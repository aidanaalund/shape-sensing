/*
 * Minimal polled driver for the TI ADS1256 24-bit delta-sigma ADC.
 * Single-ended reads against AINCOM, one channel at a time.
 */

#ifndef ADS1256_H_
#define ADS1256_H_

#include <stdint.h>

#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/spi.h>

/* PGA gain settings (ADCON[2:0]) */
enum ads1256_pga {
	ADS1256_PGA_1,
	ADS1256_PGA_2,
	ADS1256_PGA_4,
	ADS1256_PGA_8,
	ADS1256_PGA_16,
	ADS1256_PGA_32,
	ADS1256_PGA_64,
};

/* Data rates (DRATE register values, fCLKIN = 7.68 MHz) */
enum ads1256_drate {
	ADS1256_DRATE_2_5SPS = 0x03,
	ADS1256_DRATE_5SPS = 0x13,
	ADS1256_DRATE_10SPS = 0x23,
	ADS1256_DRATE_15SPS = 0x33,
	ADS1256_DRATE_25SPS = 0x43,
	ADS1256_DRATE_30SPS = 0x53,
	ADS1256_DRATE_50SPS = 0x63,
	ADS1256_DRATE_60SPS = 0x72,
	ADS1256_DRATE_100SPS = 0x82,
	ADS1256_DRATE_500SPS = 0x92,
	ADS1256_DRATE_1000SPS = 0xA1,
	ADS1256_DRATE_2000SPS = 0xB0,
};

#define ADS1256_AINCOM 8

struct ads1256 {
	struct spi_dt_spec spi;
	struct gpio_dt_spec drdy;
	uint32_t vref_mv;
	enum ads1256_pga pga;
	bool buffer;
};

#define ADS1256_DT_GET(node_id, _pga, _buffer)                                   \
	{                                                                        \
		.spi = SPI_DT_SPEC_GET(node_id, SPI_WORD_SET(8) | SPI_TRANSFER_MSB), \
		.drdy = GPIO_DT_SPEC_GET(node_id, drdy_gpios),                   \
		.vref_mv = DT_PROP(node_id, vref_mv),                            \
		.pga = (_pga),                                                   \
		.buffer = (_buffer),                                             \
	}

/** Reset, configure, self-calibrate and verify the ADC. */
int ads1256_init(const struct ads1256 *adc, enum ads1256_drate drate);

/** One settled single-ended conversion of AINx vs AINCOM (signed 24-bit code). */
int ads1256_read_raw(const struct ads1256 *adc, uint8_t ain, int32_t *raw);

/** One settled conversion with an arbitrary MUX register value (AINP << 4 | AINN). */
int ads1256_read_mux(const struct ads1256 *adc, uint8_t mux, int32_t *raw);

/** Convert a raw code to microvolts at the ADC input pin. */
int32_t ads1256_raw_to_uv(const struct ads1256 *adc, int32_t raw);

#endif /* ADS1256_H_ */
