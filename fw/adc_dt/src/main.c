/*
 * Photodiode transimpedance amplifier readout + stepper control
 * nRF52833-DK / Zephyr
 *
 *   SW1 (button0): return to 0 degrees
 *   SW2 (button1): tighten to TIGHTEN_DEG degrees
 *   SW3 (button2): debug print (readings for calibrating the photodiode)
 *   SW4 (button3): disable/reenable printing
 *
 * ADC channels (io-channels order in the overlay): Vo, Vref
 * Periodic CSV output; debug output lines start with '#' so they are easy
 * to filter out when parsing the CSV.
 */

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/adc.h>
#include <zephyr/drivers/stepper/stepper.h>
#include <zephyr/drivers/stepper/stepper_ctrl.h>
#include <zephyr/input/input.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/util.h>

/* ------------------------------------------------------------------ */
/* Tunables                                                           */
/* ------------------------------------------------------------------ */

#define SAMPLE_PERIOD_MS 1000 /* "y" Hz = 1000 / this */
#define TIGHTEN_DEG 90 /* "x": target for button 2 */
#define STEPS_PER_REV 200 /* full steps per rev (1.8 deg) */
#define STEP_INTERVAL_NS 2000000ULL /* time between micro-steps */
#define R1_OHMS 150000LL /* TIA feedback resistor */

/* ------------------------------------------------------------------ */
/* Stepper                                                            */
/* ------------------------------------------------------------------ */

static const struct device *const stepper_ctrl =
	DEVICE_DT_GET(DT_ALIAS(stepper_ctrl));

static bool streaming = true;

#define MICROSTEP_RES 1 // Edit as needed
#define STEPS_PER_360 (STEPS_PER_REV * MICROSTEP_RES)

static int32_t deg_to_steps(int32_t deg)
{
	return (int32_t)(((int64_t)deg * STEPS_PER_360) / 360);
}

static int32_t steps_to_mdeg(int32_t steps)
{
	return (int32_t)(((int64_t)steps * 360000) / STEPS_PER_360);
}

static void move_to_deg(int32_t deg)
{
	int err = stepper_ctrl_move_to(stepper_ctrl, deg_to_steps(deg));

	if (err < 0) {
		printk("# ERROR move_to %d deg (%d)\n", (int)deg, err);
	} else {
		printk("# Moving to %d deg\n", (int)deg);
	}
}

/* ------------------------------------------------------------------ */
/* Command queue (buttons + stepper callback -> main loop)            */
/* ------------------------------------------------------------------ */

enum cmd {
	CMD_RESET,
	CMD_TIGHTEN,
	CMD_DEBUG,
	CMD_MOVE_DONE,
	CMD_TOGGLE_STREAM,
};

K_MSGQ_DEFINE(cmd_q, sizeof(uint8_t), 8, 1);

static void post_cmd(enum cmd c)
{
	uint8_t v = (uint8_t)c;

	(void)k_msgq_put(&cmd_q, &v, K_NO_WAIT);
}

static void stepper_callback(const struct device *dev,
			     const enum stepper_ctrl_event event,
			     void *user_data)
{
	ARG_UNUSED(dev);
	ARG_UNUSED(user_data);

	if (event == STEPPER_CTRL_EVENT_STEPS_COMPLETED) {
		post_cmd(CMD_MOVE_DONE);
	}
}

static void button_pressed(struct input_event *evt, void *user_data)
{
	ARG_UNUSED(user_data);

	/* Only act on key presses (value 1), not releases. */
	if (evt->type != INPUT_EV_KEY || evt->value != 1) {
		return;
	}

	switch (evt->code) {
	case INPUT_KEY_0:
		post_cmd(CMD_RESET);
		break;
	case INPUT_KEY_1:
		post_cmd(CMD_TIGHTEN);
		break;
	case INPUT_KEY_2:
		post_cmd(CMD_DEBUG);
		break;
	case INPUT_KEY_3:
		post_cmd(CMD_TOGGLE_STREAM);
		break;
	default:
		break;
	}
}

/* NULL device = listen to every input device. */
INPUT_CALLBACK_DEFINE(NULL, button_pressed, NULL);

/* ------------------------------------------------------------------ */
/* ADC                                                                */
/* ------------------------------------------------------------------ */

#define VO_IDX    0
#define VREF_IDX  1
#define NUM_CH    2

#define DT_SPEC_AND_COMMA(node_id, prop, idx) \
	ADC_DT_SPEC_GET_BY_IDX(node_id, idx),

/* io-channels order in the overlay: Vo, Vref */
static const struct adc_dt_spec adc_channels[] = {
	DT_FOREACH_PROP_ELEM(DT_PATH(zephyr_user), io_channels,
			     DT_SPEC_AND_COMMA)
};

BUILD_ASSERT(ARRAY_SIZE(adc_channels) == NUM_CH,
	     "Expected 2 ADC channels in io-channels: Vo, Vref");

static int16_t adc_buf[NUM_CH];

struct measurement {
	int16_t raw[NUM_CH];
	int32_t mv[NUM_CH];
	int64_t i_pa;       /* photodiode current in picoamps */
};

static int measure(struct adc_sequence *seq, struct measurement *m)
{
	int err = adc_read_dt(&adc_channels[0], seq);

	if (err < 0) {
		printk("# ERROR adc read (%d)\n", err);
		return err;
	}
	seq->calibrate = false; /* calibrate only on the first read */

	for (size_t i = 0; i < NUM_CH; i++) {
		m->raw[i] = adc_buf[i];
		m->mv[i] = adc_buf[i];
		err = adc_raw_to_millivolts_dt(&adc_channels[i], &m->mv[i]);
		if (err < 0) {
			printk("# ERROR mV conversion ch%d (%d)\n", (int)i, err);
			return err;
		}
	}

	/* mV / ohm -> pA:  (mV * 1e-3 / R) * 1e12 = mV * 1e9 / R */
	m->i_pa = ((int64_t)(m->mv[VO_IDX] - m->mv[VREF_IDX]) * 1000000000LL) /
		  R1_OHMS;

	return 0;
}

/* ------------------------------------------------------------------ */
/* Output                                                             */
/* ------------------------------------------------------------------ */

static void print_csv(int64_t t_ms, const struct measurement *m)
{
	int32_t pos = 0;

	(void)stepper_ctrl_get_actual_position(stepper_ctrl, &pos);

	printk("%" PRId64 ",%" PRId32 ",%" PRId32 ",%" PRId64 ",%" PRId32
	       ",%" PRId32 "\n",
	       t_ms, m->mv[VO_IDX], m->mv[VREF_IDX], m->i_pa, pos,
	       steps_to_mdeg(pos));
}

static void print_debug(const struct measurement *m)
{
	int32_t pos = 0;

	(void)stepper_ctrl_get_actual_position(stepper_ctrl, &pos);

	printk("# DEBUG Vo: raw=%d  %d mV | Vref: raw=%d  %d mV | "
	       "Vo-Vref=%d mV | I=%" PRId64 " pA | pos=%d steps (%d mdeg)\n",
	       m->raw[VO_IDX], (int)m->mv[VO_IDX],
	       m->raw[VREF_IDX], (int)m->mv[VREF_IDX],
	       (int)(m->mv[VO_IDX] - m->mv[VREF_IDX]), m->i_pa,
	       (int)pos, (int)steps_to_mdeg(pos));
}

/* ------------------------------------------------------------------ */
/* Setup helpers                                                      */
/* ------------------------------------------------------------------ */

static int stepper_setup(void)
{
	int err;

	if (!device_is_ready(stepper_ctrl)) {
		printk("# ERROR stepper device not ready\n");
		return -ENODEV;
	}

	err = stepper_ctrl_set_event_cb(stepper_ctrl, stepper_callback, NULL);
	if (err < 0) {
		printk("# ERROR set_event_cb (%d)\n", err);
		return err;
	}

	/* Whatever position the motor is in at boot is defined as 0 deg. */
	(void)stepper_ctrl_set_reference_position(stepper_ctrl, 0);
	(void)stepper_ctrl_set_microstep_interval(stepper_ctrl, STEP_INTERVAL_NS);

	return 0;
}

static int adc_setup(struct adc_sequence *seq)
{
	int err;

	for (size_t i = 0; i < NUM_CH; i++) {
		if (!adc_is_ready_dt(&adc_channels[i])) {
			printk("# ERROR ADC %s not ready\n",
			       adc_channels[i].dev->name);
			return -ENODEV;
		}

		err = adc_channel_setup_dt(&adc_channels[i]);
		if (err < 0) {
			printk("# ERROR channel setup #%d (%d)\n", (int)i, err);
			return err;
		}

		/* Samples land in the buffer in ascending channel-id order,
		 * so channel ids must match the io-channels order. */
		if (adc_channels[i].channel_id != i) {
			printk("# ERROR channel ids must be 0..%d in order\n",
			       NUM_CH - 1);
			return -EINVAL;
		}
	}

	/* One sequence covering all channels. Resolution and oversampling are
	 * taken from channel 0, so both channels should use the same values. */
	err = adc_sequence_init_dt(&adc_channels[0], seq);
	if (err < 0) {
		return err;
	}
	for (size_t i = 1; i < NUM_CH; i++) {
		seq->channels |= BIT(adc_channels[i].channel_id);
	}

	return 0;
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

static void handle_cmd(enum cmd c, struct adc_sequence *seq)
{
	struct measurement m;

	switch (c) {
		case CMD_RESET:
			move_to_deg(0);
			break;
		case CMD_TIGHTEN:
			move_to_deg(TIGHTEN_DEG);
			break;
		case CMD_DEBUG:
			if (measure(seq, &m) == 0) {
				print_debug(&m);
			}
			break;
		case CMD_MOVE_DONE: {
			int32_t pos = 0;
			(void)stepper_ctrl_get_actual_position(stepper_ctrl, &pos);
			printk("# Move complete: pos=%d steps (%d mdeg)\n",
				(int)pos, (int)steps_to_mdeg(pos));
			break;
		}
		case CMD_TOGGLE_STREAM:
			streaming = !streaming;
			if (streaming) {
				printk("t_ms,vo_mv,vref_mv,i_pa,pos_steps,pos_mdeg\n");
			} else {
				printk("# Streaming stopped\n");
			}
			break;
	}
}

int main(void)
{
	struct adc_sequence sequence = {
		.buffer = adc_buf,
		.buffer_size = sizeof(adc_buf),
		.calibrate = true,
	};

	if (stepper_setup() < 0 || adc_setup(&sequence) < 0) {
		return 0;
	}

	printk("t_ms,vo_mv,vref_mv,i_pa,pos_steps,pos_mdeg\n");

	int64_t next_ms = k_uptime_get();

	while (1) {
		uint8_t c;

		/* Waiting for a command doubles as the sampling timer: if no
		 * command arrives before next_ms, we take a sample. Stepper
		 * moves are non-blocking, so sampling continues during motion. */
		if (k_msgq_get(&cmd_q, &c, K_TIMEOUT_ABS_MS(next_ms)) == 0) {
			handle_cmd((enum cmd)c, &sequence);
			continue;
		}

		if (streaming) {
			struct measurement m;
			int64_t now = k_uptime_get();

			if (measure(&sequence, &m) == 0) {
				print_csv(now, &m);
			}
		}
		next_ms += SAMPLE_PERIOD_MS;
	}

	return 0;
}