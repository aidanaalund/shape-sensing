#include "ads1256.h"

#include <errno.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/util.h>

/* Commands */
#define CMD_WAKEUP 0x00
#define CMD_RDATA  0x01
#define CMD_SDATAC 0x0F
#define CMD_RREG   0x10
#define CMD_WREG   0x50
#define CMD_SELFCAL 0xF0
#define CMD_SYNC   0xFC
#define CMD_RESET  0xFE

/* Registers */
#define REG_STATUS 0x00
#define REG_MUX    0x01
#define REG_ADCON  0x02
#define REG_DRATE  0x03

#define STATUS_BUFEN BIT(1)

/* Timing (fCLKIN = 7.68 MHz): t6 = 50 tCLKIN = 6.5 us, t11 >= 24 tCLKIN = 3.1 us */
#define T6_US  10
#define T11_US 5

#define DRDY_TIMEOUT_MS 2000

static int wait_drdy(const struct ads1256 *adc)
{
	int64_t end = k_uptime_get() + DRDY_TIMEOUT_MS;

	while (gpio_pin_get_dt(&adc->drdy) != 1) {
		if (k_uptime_get() >= end) {
			return -ETIMEDOUT;
		}
		k_usleep(200);
	}
	return 0;
}

/* After RESET/SELFCAL, DRDY may still be low from the previous conversion:
 * give it a moment to go high, then wait for it to fall again. */
static int wait_drdy_fresh(const struct ads1256 *adc)
{
	for (int i = 0; i < 50 && gpio_pin_get_dt(&adc->drdy) == 1; i++) {
		k_usleep(200);
	}
	return wait_drdy(adc);
}

static int cmd_write(const struct ads1256 *adc, const uint8_t *tx, size_t len)
{
	const struct spi_buf buf = {.buf = (void *)tx, .len = len};
	const struct spi_buf_set set = {.buffers = &buf, .count = 1};

	return spi_write_dt(&adc->spi, &set);
}

/* Command bytes, wait t6, then read: CS is held low for the whole sequence. */
static int cmd_read(const struct ads1256 *adc, const uint8_t *cmd, size_t cmd_len,
		    uint8_t *rx, size_t rx_len)
{
	struct spi_config cfg = adc->spi.config;
	const struct spi_buf tx_buf = {.buf = (void *)cmd, .len = cmd_len};
	const struct spi_buf_set tx_set = {.buffers = &tx_buf, .count = 1};
	const struct spi_buf rx_buf = {.buf = rx, .len = rx_len};
	const struct spi_buf_set rx_set = {.buffers = &rx_buf, .count = 1};
	int err;

	cfg.operation |= SPI_HOLD_ON_CS | SPI_LOCK_ON;

	err = spi_write(adc->spi.bus, &cfg, &tx_set);
	if (err == 0) {
		k_busy_wait(T6_US);
		err = spi_read(adc->spi.bus, &cfg, &rx_set);
	}
	spi_release(adc->spi.bus, &cfg);

	return err;
}

static int write_regs(const struct ads1256 *adc, uint8_t reg, const uint8_t *val,
		      uint8_t n)
{
	uint8_t tx[2 + 4] = {CMD_WREG | reg, n - 1};

	memcpy(&tx[2], val, n);
	return cmd_write(adc, tx, 2 + n);
}

static int read_regs(const struct ads1256 *adc, uint8_t reg, uint8_t *val, uint8_t n)
{
	const uint8_t cmd[] = {CMD_RREG | reg, n - 1};

	return cmd_read(adc, cmd, sizeof(cmd), val, n);
}

static int cmd_byte(const struct ads1256 *adc, uint8_t cmd)
{
	return cmd_write(adc, &cmd, 1);
}

int ads1256_init(const struct ads1256 *adc, enum ads1256_drate drate)
{
	uint8_t regs[4], rb[4];
	int err;

	if (!spi_is_ready_dt(&adc->spi) || !gpio_is_ready_dt(&adc->drdy)) {
		return -ENODEV;
	}

	err = gpio_pin_configure_dt(&adc->drdy, GPIO_INPUT);
	if (err < 0) {
		return err;
	}

	k_msleep(100); /* power-up / oscillator start */

	/* Reset, then wait for the power-up self-calibration (DRDY low). */
	err = cmd_byte(adc, CMD_RESET);
	if (err < 0) {
		return err;
	}
	err = wait_drdy_fresh(adc);
	if (err < 0) {
		printk("# ERROR ADS1256 no DRDY after reset (check wiring/power)\n");
		return err;
	}

	err = cmd_byte(adc, CMD_SDATAC);
	if (err < 0) {
		return err;
	}
	k_busy_wait(T11_US);

	regs[0] = adc->buffer ? STATUS_BUFEN : 0; /* MSB first, no auto-cal */
	regs[1] = (0 << 4) | ADS1256_AINCOM;      /* AIN0 vs AINCOM */
	regs[2] = adc->pga;                       /* CLKOUT off, sensor detect off */
	regs[3] = drate;
	err = write_regs(adc, REG_STATUS, regs, 4);
	if (err < 0) {
		return err;
	}

	err = read_regs(adc, REG_STATUS, rb, 4);
	if (err < 0) {
		return err;
	}
	uint8_t id = rb[0] >> 4;

	/* STATUS[0] is DRDY and may differ; ID must be non-trivial (bus alive). */
	if (id == 0x0 || id == 0xF || (rb[0] & 0x0E) != regs[0] || rb[1] != regs[1] ||
	    rb[2] != regs[2] || rb[3] != regs[3]) {
		printk("# ERROR ADS1256 readback mismatch: %02x %02x %02x %02x\n", rb[0],
		       rb[1], rb[2], rb[3]);
		return -EIO;
	}
	printk("# ADS1256 id=%d\n", id);

	/* Offset/gain self-calibration with the final PGA/buffer/data-rate settings. */
	err = cmd_byte(adc, CMD_SELFCAL);
	if (err < 0) {
		return err;
	}
	return wait_drdy_fresh(adc);
}

int ads1256_read_raw(const struct ads1256 *adc, uint8_t ain, int32_t *raw)
{
	if (ain > 7) {
		return -EINVAL;
	}

	return ads1256_read_mux(adc, (ain << 4) | ADS1256_AINCOM, raw);
}

int ads1256_read_mux(const struct ads1256 *adc, uint8_t mux, int32_t *raw)
{
	uint8_t d[3];
	int err;

	/* Datasheet "cycling the input multiplexer": WREG MUX, SYNC, WAKEUP, wait
	 * for DRDY (stays high until the new, fully settled conversion), RDATA. */
	err = write_regs(adc, REG_MUX, &mux, 1);
	if (err < 0) {
		return err;
	}
	k_busy_wait(T11_US);
	err = cmd_byte(adc, CMD_SYNC);
	if (err < 0) {
		return err;
	}
	k_busy_wait(T11_US);
	err = cmd_byte(adc, CMD_WAKEUP);
	if (err < 0) {
		return err;
	}

	err = wait_drdy(adc);
	if (err < 0) {
		return err;
	}

	err = cmd_read(adc, (const uint8_t[]){CMD_RDATA}, 1, d, sizeof(d));
	if (err < 0) {
		return err;
	}

	/* Sign-extend the 24-bit two's complement result. */
	*raw = (int32_t)(((uint32_t)d[0] << 24) | ((uint32_t)d[1] << 16) |
			 ((uint32_t)d[2] << 8)) >> 8;
	return 0;
}

int32_t ads1256_raw_to_uv(const struct ads1256 *adc, int32_t raw)
{
	/* Full scale is +/-2*Vref/PGA for +/-(2^23 - 1) */
	int64_t fs_uv = 2LL * adc->vref_mv * 1000 / (1 << adc->pga);

	return (int32_t)(((int64_t)raw * fs_uv) / 0x7FFFFF);
}
