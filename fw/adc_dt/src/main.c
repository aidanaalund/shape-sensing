/*
 * Photodiode transimpedance amplifier readout (3x Vo + shared Vref)
 */

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/adc.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/util.h>

#define DT_SPEC_AND_COMMA(node_id, prop, idx) \
	ADC_DT_SPEC_GET_BY_IDX(node_id, idx),

/* io-channels order in the overlay: Vo1, Vo2, Vo3, Vref */
static const struct adc_dt_spec adc_channels[] = {
	DT_FOREACH_PROP_ELEM(DT_PATH(zephyr_user), io_channels,
			     DT_SPEC_AND_COMMA)
};

#define NUM_PD        3                    /* number of photodiodes */
#define VREF_IDX      NUM_PD               /* index of the Vref channel */
#define NUM_CH        ARRAY_SIZE(adc_channels)

BUILD_ASSERT(ARRAY_SIZE(adc_channels) == NUM_PD + 1,
	     "Expected 3 photodiode channels + 1 Vref channel");

/* Feedback resistor per photodiode, in ohms. SET THESE TO YOUR REAL VALUES. */
static const int64_t r1_ohms[NUM_PD] = {
	2000000, 2000000, 2000000,
};

#define SAMPLE_PERIOD_MS 1000

static int16_t buf[NUM_CH];

static int sample_once(struct adc_sequence *seq, int64_t t_ms)
{
	int err = adc_read_dt(&adc_channels[0], seq);

	if (err < 0) {
		printk("# ERROR read (%d)\n", err);
		return err;
	}
	seq->calibrate = false; /* only after a successful first read */

	int32_t mv[NUM_CH];

	for (size_t i = 0U; i < NUM_CH; i++) {
		mv[i] = (int32_t)buf[i];
		err = adc_raw_to_millivolts_dt(&adc_channels[i], &mv[i]);
		if (err < 0) {
			printk("# ERROR mV conversion ch%d (%d)\n", (int)i, err);
			return err;
		}
	}

	int64_t i_pa[NUM_PD];

	for (size_t i = 0U; i < NUM_PD; i++) {
		i_pa[i] = ((int64_t)(mv[i] - mv[VREF_IDX]) * 1000000000LL) / r1_ohms[i];
	}

	printk("%" PRId64 ",%" PRId32 ",%" PRId32 ",%" PRId32 ",%" PRId32
	       ",%" PRId64 ",%" PRId64 ",%" PRId64 "\n",
	       t_ms, mv[0], mv[1], mv[2], mv[VREF_IDX],
	       i_pa[0], i_pa[1], i_pa[2]);

	return 0;
}

int main(void)
{
	int err;
	struct adc_sequence sequence = {
		.buffer = buf, /* in bytes */
		.buffer_size = sizeof(buf),
		.calibrate = true, /* calibrate on the first read only */
	};

	/* Configure channels individually prior to sampling. */
	for (size_t i = 0U; i < NUM_CH; i++) {
		if (!adc_is_ready_dt(&adc_channels[i])) {
			printk("# ERROR ADC %s not ready\n", adc_channels[i].dev->name);
			return 0;
		}

		err = adc_channel_setup_dt(&adc_channels[i]);
		if (err < 0) {
			printk("# ERROR channel setup #%d (%d)\n", (int)i, err);
			return 0;
		}

		/* Samples land in the buffer in ascending channel-id order,
		 * so channel ids must match the io-channels order. */
		if (adc_channels[i].channel_id != i) {
			printk("# ERROR channel ids must be 0..%d in order\n",
			       (int)NUM_CH - 1);
			return 0;
		}
	}

	/* One sequence covering all channels (all share resolution/oversampling) */
	(void)adc_sequence_init_dt(&adc_channels[0], &sequence);
	for (size_t i = 1U; i < NUM_CH; i++) {
		sequence.channels |= BIT(adc_channels[i].channel_id);
	}

	printk("t_ms,vo1_mv,vo2_mv,vo3_mv,vref_mv,i1_pa,i2_pa,i3_pa\n");

		int64_t next_ms = k_uptime_get();

	while (1) {
		(void)sample_once(&sequence, k_uptime_get());
		next_ms += SAMPLE_PERIOD_MS;
		k_sleep(K_TIMEOUT_ABS_MS(next_ms));
	}
	return 0;
}