/*
 * Photodiode transimpedance amplifier readout + stepper control
 * XIAO ESP32-S3 + ADS1256 / Zephyr
 *
 * Stepper control is over the USB serial console (type in any serial
 * terminal). Input is only accepted while the stepper is idle; anything
 * typed while it is moving is discarded.
 *
 *   0..359 <Enter>  move to that absolute angle (degrees)
 *   ] / [           increment / decrement by MOVE_SMALL (1 deg)
 *   } / {           increment / decrement by MOVE_MED (10 deg)
 *   > / <           increment / decrement by MOVE_LARGE (90 deg)
 *   p               pause / resume CSV streaming
 *   d               debug print (raw readings, for calibrating the photodiodes)
 *   ?               help
 *
 * ADS1256 channels (single-ended vs AINCOM):
 *	AIN0 = Vref, AIN1..AIN7 = Vo1..Vo7 (seven photodiode amplifier outputs)
 *
 * Periodic CSV output; all other output lines start with '#' so they are easy
 * to filter out when parsing the CSV.
 */

#include <errno.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/stepper/stepper.h>
#include <zephyr/drivers/stepper/stepper_ctrl.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/util.h>

#include "ads1256.h"

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */

#define DIV_NUM 25  /* Output voltage divider (10k + 15k) */
#define DIV_DEN 15  /* Output voltage divider top (15k) */
#define R1_OHMS 150000LL /* Feedback resistor */

#define SAMPLE_PERIOD_MS 500 /* "y" Hz = 1000 / this */
#define ADS_DRATE ADS1256_DRATE_60SPS
#define ADS_PGA ADS1256_PGA_1
#define ADS_BUFFER false
#define MOVE_LARGE 90
#define MOVE_MED 10
#define MOVE_SMALL 1
#define MAX_DEG 359
#define STEPS_PER_REV 200 /* full steps per rev (1.8 deg) */
#define STEP_INTERVAL_NS 500000000ULL /* time between micro-steps */

#define INPUT_POLL_MS 5

/* ------------------------------------------------------------------ */
/* Stepper                                                            */
/* ------------------------------------------------------------------ */

static const struct device *const stepper_ctrl =
	DEVICE_DT_GET(DT_ALIAS(stepper_ctrl));

static bool streaming = true;

/* Set when a move starts, cleared when the stepper reports completion. */
static atomic_t moving;

/* Last commanded angle. Tracked separately from the motor position because
 * one step is 1.8 deg: small increments accumulate here and only turn into a
 * physical step once they add up. */
static int32_t cmd_deg;

#define MICROSTEP_RES 1 // Not 1 degree resolution (change to 8 to achieve, but rewire MS pins)
#define STEPS_PER_360 (STEPS_PER_REV * MICROSTEP_RES)

/* Nearest step (round half away from zero). */
static int32_t deg_to_steps(int32_t deg)
{
	int32_t n = deg * STEPS_PER_360;

	return (n >= 0 ? n + 180 : n - 180) / 360;
}

static int32_t steps_to_mdeg(int32_t steps)
{
	return (int32_t)(((int64_t)steps * 360000) / STEPS_PER_360);
}

static void move_to_deg(int32_t deg)
{
	int32_t pos = 0;
	int32_t target = deg_to_steps(deg);
	int err;

	cmd_deg = deg;

	(void)stepper_ctrl_get_actual_position(stepper_ctrl, &pos);
	if (target == pos) {
		printk("# At %d deg (no step needed)\n", (int)deg);
		return;
	}

	atomic_set(&moving, 1);
	err = stepper_ctrl_move_to(stepper_ctrl, target);
	if (err < 0) {
		atomic_set(&moving, 0);
		printk("# ERROR move_to %d deg (%d)\n", (int)deg, err);
	} else {
		printk("# Moving to %d deg (%d steps)\n", (int)deg, (int)target);
	}
}

/* ------------------------------------------------------------------ */
/* Command queue (serial input + stepper callback -> main)           */
/* ------------------------------------------------------------------ */

enum cmd {
	CMD_GOTO,	/* arg = absolute degrees */
	CMD_STEP,	/* arg = signed delta in degrees */
	CMD_DEBUG,
	CMD_MOVE_DONE,
	CMD_TOGGLE_STREAM,
};

struct cmd_msg {
	uint8_t cmd;
	int16_t arg;
};

K_MSGQ_DEFINE(cmd_q, sizeof(struct cmd_msg), 8, 2);

static void post_cmd(enum cmd c, int16_t arg)
{
	struct cmd_msg m = {.cmd = (uint8_t)c, .arg = arg};

	(void)k_msgq_put(&cmd_q, &m, K_NO_WAIT);
}

static void stepper_callback(const struct device *dev,
			     const enum stepper_ctrl_event event,
			     void *user_data)
{
	ARG_UNUSED(dev);
	ARG_UNUSED(user_data);

	if (event == STEPPER_CTRL_EVENT_STEPS_COMPLETED) {
		post_cmd(CMD_MOVE_DONE, 0);
	}
}

/* ------------------------------------------------------------------ */
/* Serial input (USB console -> commands)                             */
/* ------------------------------------------------------------------ */

static const struct device *const console_dev =
	DEVICE_DT_GET(DT_CHOSEN(zephyr_console));

static void print_help(void)
{
	printk("# Commands (only accepted while the stepper is idle):\n"
	       "#   0-%d <Enter>  go to absolute angle (deg)\n"
	       "#   ] / [        +/- %d deg\n"
	       "#   } / {        +/- %d deg\n"
	       "#   > / <        +/- %d deg\n"
	       "#   p            pause/resume CSV streaming\n"
	       "#   d            debug print\n",
	       MAX_DEG, MOVE_SMALL, MOVE_MED, MOVE_LARGE);
}

/* Returns the signed step for a one-key increment command, or 0. */
static int16_t key_to_delta(unsigned char ch)
{
	switch (ch) {
	case ']': return MOVE_SMALL;
	case '[': return -MOVE_SMALL;
	case '}': return MOVE_MED;
	case '{': return -MOVE_MED;
	case '>': return MOVE_LARGE;
	case '<': return -MOVE_LARGE;
	default: return 0;
	}
}

static void input_thread(void *a, void *b, void *c)
{
	char num[4];	/* up to 3 digits + NUL */
	size_t n = 0;

	ARG_UNUSED(a);
	ARG_UNUSED(b);
	ARG_UNUSED(c);

	while (1) {
		unsigned char ch;

		if (uart_poll_in(console_dev, &ch) != 0) {
			k_sleep(K_MSEC(INPUT_POLL_MS));
			continue;
		}

		if (ch == '\n' && n == 0) {
			continue; /* second half of a CRLF */
		}

		/* The stepper is busy: discard the input, and any half-typed number. */
		if (atomic_get(&moving)) {
			n = 0;
			if (ch == '\r' || ch == '\n' || ch == '?' || key_to_delta(ch) != 0) {
				printk("# Busy, input ignored\n");
			}
			if (ch != 'p' && ch != 'd') {
				continue;
			}
			/* 'p' and 'd' do not move the stepper, so they stay available. */
		}

		if (ch >= '0' && ch <= '9') {
			if (n < sizeof(num) - 1) {
				num[n++] = (char)ch;
				printk("%c", ch);
			}
		} else if (ch == '\r' || ch == '\n') {
			if (n == 0) {
				continue;
			}
			num[n] = '\0';
			n = 0;

			int v = 0;

			for (const char *s = num; *s; s++) {
				v = v * 10 + (*s - '0');
			}
			printk("\n");
			if (v > MAX_DEG) {
				printk("# ERROR %d is out of range [0,%d]\n", v, MAX_DEG);
			} else {
				post_cmd(CMD_GOTO, (int16_t)v);
			}
		} else if (ch == 0x08 || ch == 0x7f) {
			if (n > 0) {
				n--;
				printk("\b \b");
			}
		} else if (ch == 'p') {
			n = 0;
			post_cmd(CMD_TOGGLE_STREAM, 0);
		} else if (ch == 'd') {
			n = 0;
			post_cmd(CMD_DEBUG, 0);
		} else if (ch == '?') {
			n = 0;
			print_help();
		} else if (key_to_delta(ch) != 0) {
			n = 0;
			post_cmd(CMD_STEP, key_to_delta(ch));
		} else {
			n = 0; /* unknown key: drop any partial number */
		}
	}
}

K_THREAD_DEFINE(input_tid, 1024, input_thread, NULL, NULL, NULL, 7, 0, 0);

/* ------------------------------------------------------------------ */
/* ADC                                                                */
/* ------------------------------------------------------------------ */

#define VREF_IDX  0
#define NUM_SIG   7			/* photodiode channels */
#define NUM_CH    (1 + NUM_SIG)	/* Vref + signals; channel i is AINi */

static const struct ads1256 adc =
	ADS1256_DT_GET(DT_NODELABEL(ads1256), ADS_PGA, ADS_BUFFER);

struct measurement {
	int32_t raw[NUM_CH];
	int32_t uv[NUM_CH];
	int64_t i_pa[NUM_SIG];	/* photodiode currents in picoamps */
};

static int measure(struct measurement *m)
{
	for (size_t i = 0; i < NUM_CH; i++) {
		int err = ads1256_read_raw(&adc, (uint8_t)i, &m->raw[i]);

		if (err < 0) {
			printk("# ERROR adc read AIN%d (%d)\n", (int)i, err);
			return err;
		}
		m->uv[i] = ads1256_raw_to_uv(&adc, m->raw[i]);
	}

	for (size_t s = 0; s < NUM_SIG; s++) {
		/* uV / ohm = uA = 1e6 pA */
		int64_t vo_amp_uv = (int64_t)m->uv[1 + s] * DIV_NUM / DIV_DEN;

		m->i_pa[s] = ((vo_amp_uv - m->uv[VREF_IDX]) * 1000000LL) / R1_OHMS;
	}

	return 0;
}

/* ------------------------------------------------------------------ */

static void print_csv_header(void)
{
	printk("t_ms,vref_uv");
	for (int i = 1; i <= NUM_SIG; i++) {
		printk(",vo%d_uv", i);
	}
	for (int i = 1; i <= NUM_SIG; i++) {
		printk(",i%d_pa", i);
	}
	printk(",pos_steps,pos_mdeg\n");
}

static void print_csv(int64_t t_ms, const struct measurement *m)
{
	int32_t pos = 0;

	(void)stepper_ctrl_get_actual_position(stepper_ctrl, &pos);

	printk("%" PRId64 ",%" PRId32, t_ms, m->uv[VREF_IDX]);
	for (size_t i = 1; i < NUM_CH; i++) {
		printk(",%" PRId32, m->uv[i]);
	}
	for (size_t s = 0; s < NUM_SIG; s++) {
		printk(",%" PRId64, m->i_pa[s]);
	}
	printk(",%" PRId32 ",%" PRId32 "\n", pos, steps_to_mdeg(pos));
}

static void print_debug(const struct measurement *m)
{
	int32_t pos = 0;

	(void)stepper_ctrl_get_actual_position(stepper_ctrl, &pos);

	printk("# DEBUG Vref: raw=%d  %d uV\n", (int)m->raw[VREF_IDX],
	       (int)m->uv[VREF_IDX]);
	for (size_t s = 0; s < NUM_SIG; s++) {
		printk("# DEBUG Vo%d: raw=%d  %d uV | Vo-Vref=%d uV | "
		       "I=%" PRId64 " pA\n",
		       (int)(s + 1), (int)m->raw[1 + s], (int)m->uv[1 + s],
		       (int)(m->uv[1 + s] - m->uv[VREF_IDX]), m->i_pa[s]);
	}
	printk("# DEBUG pos=%d steps (%d mdeg)\n", (int)pos,
	       (int)steps_to_mdeg(pos));
}

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
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
	err = stepper_ctrl_set_reference_position(stepper_ctrl, 0);
	if (err < 0) {
		printk("# ERROR set_reference_position (%d)\n", err);
		return err;
	}
	err = stepper_ctrl_set_microstep_interval(stepper_ctrl, STEP_INTERVAL_NS);
	if (err < 0) {
		printk("# ERROR set_microstep_interval (%d)\n", err);
		return err;
	}

	return 0;
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

static void handle_debug(void)
{
	struct measurement m;

	if (measure(&m) == 0) {
		print_debug(&m);
	}
}

static void handle_cmd(const struct cmd_msg *msg)
{
	/* Motion commands are only accepted while the stepper is idle. */
	if ((msg->cmd == CMD_GOTO || msg->cmd == CMD_STEP) &&
	    atomic_get(&moving)) {
		printk("# Busy, command ignored\n");
		return;
	}

	switch (msg->cmd) {
	case CMD_GOTO:
		move_to_deg(msg->arg);
		break;
	case CMD_STEP: {
		int32_t target = CLAMP(cmd_deg + msg->arg, 0, MAX_DEG);

		if (target == cmd_deg) {
			printk("# At limit (%d deg)\n", (int)cmd_deg);
		} else {
			move_to_deg(target);
		}
		break;
	}
	case CMD_DEBUG:
		handle_debug();
		break;
	case CMD_MOVE_DONE: {
		int32_t pos = 0;

		atomic_set(&moving, 0);
		(void)stepper_ctrl_get_actual_position(stepper_ctrl, &pos);
		printk("# Move complete: pos=%d steps (%d mdeg), ready for input\n",
		       (int)pos, (int)steps_to_mdeg(pos));
		break;
	}
	case CMD_TOGGLE_STREAM:
		streaming = !streaming;
		if (streaming) {
			print_csv_header();
		} else {
			printk("# Streaming stopped\n");
		}
		break;
	}
}

int main(void)
{
	int err;

	if (!device_is_ready(console_dev)) {
		return 0;
	}

	err = stepper_setup();
	if (err < 0) {
		return 0;
	}

	err = ads1256_init(&adc, ADS_DRATE);
	if (err < 0) {
		printk("# ERROR ADS1256 init (%d)\n", err);
		return 0;
	}

	print_help();
	print_csv_header();

	int64_t next_ms = k_uptime_get();

	while (1) {
		struct cmd_msg msg;

		/* Waiting for a command doubles as the sampling timer: if no
		 * command arrives before next_ms, we take a sample. Stepper
		 * moves are non-blocking, so sampling continues during motion. */
		if (k_msgq_get(&cmd_q, &msg, K_TIMEOUT_ABS_MS(next_ms)) == 0) {
			handle_cmd(&msg);
			continue;
		}

		if (streaming) {
			struct measurement m;
			int64_t now = k_uptime_get();

			if (measure(&m) == 0) {
				print_csv(now, &m);
			}
		}

		// A slow read or a long command must not cause a burst of back-to-back samples afterwards.
		next_ms += SAMPLE_PERIOD_MS;
		if (next_ms < k_uptime_get()) {
			next_ms = k_uptime_get() + SAMPLE_PERIOD_MS;
		}
	}

	return 0;
}
