// apt_top.v - Artificial Peripheral Thalamus, FPGA top level.
//
//   sensors ──SPI ADC──┐                       ┌──> H-bridge (centre-aligned PWM, dead-time)
//   encoder ──quad─────┼──> apt_core (1 kHz) ──┼──> 4 x biphasic stimulator (afferent return)
//   e-stop ──sync──────┘     │      ▲          └──> telemetry (SLIP/CRC over UART)
//                            │      └── CSRs <── SPI slave <── supervisory MCU (fw/)
//                            └──> latency monitor (sample->PWM, sample->stimulus)
//
// Everything the datapath computes is specified by include/apt/core_model.hpp and verified
// against it byte for byte by tb/tb_apt_top.v.
`default_nettype none

module apt_top #(
    parameter integer TICK_DIV  = 50000,  // clk cycles per control tick (50 MHz / 1 kHz)
    parameter integer UART_DIVP = 20
) (
    input  wire        clk,
    input  wire        rst_n,
    // 8-channel simultaneous-sampling ADC
    output wire        adc_convst,
    input  wire        adc_busy,
    output wire        adc_cs_n,
    output wire        adc_sclk,
    input  wire        adc_dout,
    // linear encoder
    input  wire        enc_a,
    input  wire        enc_b,
    // H-bridge gate drives
    output wire        pwm_ha,
    output wire        pwm_la,
    output wire        pwm_hb,
    output wire        pwm_lb,
    // afferent stimulator, 4 channels
    output wire [3:0]  stim_cath,
    output wire [3:0]  stim_anod,
    output wire [31:0] stim_dac,
    // safety
    input  wire        estop_n,
    output wire        fault_led,
    output wire        sample_o,
    // supervisory MCU
    input  wire        spi_sclk,
    input  wire        spi_cs_n,
    input  wire        spi_mosi,
    output wire        spi_miso,
    // telemetry
    output wire        tlm_tx
);
`include "apt_regmap.vh"
`include "apt_params.vh"

    // ------------------------------------------------------------------ reset
    reg [1:0] rst_s;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) rst_s <= 2'b11;
        else        rst_s <= {rst_s[0], 1'b0};
    wire rst = rst_s[1];

    // ------------------------------------------------------------------ CSR file
    reg [31:0] mem [0:127];
`include "apt_csr.vh"

    wire [6:0]  csr_addr;
    wire        csr_rd, csr_wr;
    wire [31:0] csr_wdata;
    reg  [31:0] csr_rdata;
    reg         fclr_pend, hb_pend;

    apt_spi_slave u_spi (
        .clk(clk), .rst(rst), .sclk(spi_sclk), .cs_n(spi_cs_n), .mosi(spi_mosi), .miso(spi_miso),
        .addr(csr_addr), .rd(csr_rd), .rdata(csr_rdata), .wr(csr_wr), .wdata(csr_wdata)
    );

    wire [31:0] ctrl = mem[A_CTRL];
    wire        run  = ctrl[4];

    // ------------------------------------------------------------------ tick generator
    reg [31:0] tick_div;
    reg        tick;
    always @(posedge clk) begin
        tick <= 1'b0;
        if (rst || !run) tick_div <= 32'd0;
        else if (tick_div == 32'd0) begin
            tick <= 1'b1;
            tick_div <= TICK_DIV - 1;
        end else tick_div <= tick_div - 32'd1;
    end
    assign sample_o = adc_convst;

    // ------------------------------------------------------------------ sensors
    wire [127:0] codes;
    wire         adc_done, adc_fault;
    apt_spi_adc #(.NCH(8), .SCLK_HALF(ADC_SCLK_HALF), .CONV(ADC_CONV_CYC), .TIMEOUT(ADC_TIMEOUT)) u_adc (
        .clk(clk), .rst(rst), .start(tick), .convst(adc_convst), .busy_in(adc_busy),
        .cs_n(adc_cs_n), .sclk(adc_sclk), .dout(adc_dout), .done(adc_done), .fault(adc_fault),
        .codes(codes)
    );

    wire signed [31:0] enc_count;
    wire [31:0]        enc_errors;
    apt_quad u_quad (.clk(clk), .rst(rst), .a_in(enc_a), .b_in(enc_b), .count(enc_count), .errors(enc_errors));

    reg [1:0] estop_s;
    always @(posedge clk) estop_s <= {estop_s[0], ~estop_n};
    wire estop_now = estop_s[1];

    // sampled at the sampling instant, like the ADC
    reg signed [31:0] enc_at_sample;
    reg               estop_at_sample;
    always @(posedge clk) if (tick) begin enc_at_sample <= enc_count; estop_at_sample <= estop_now; end

    // ------------------------------------------------------------------ datapath
    wire        core_done, core_active, core_fault, core_busy;
    wire [31:0] tick_tlm, stim_amp_k, spike_count, pwm_cmp;
    wire [255:0] filt_flat;
    wire signed [31:0] F, x, v, x_cmd, v_cmd, e, duty;
    wire [3:0]  spikes, gated;
    wire [15:0] status, rep_count;
    wire        pwm_rev;
    wire        lat_overrun;
    reg         latf_pend;

    apt_core #(.PWM_PERIOD(PWM_PERIOD)) u_core (
        .clk(clk), .rst(rst), .run(run), .start(adc_done), .codes(codes), .enc(enc_at_sample),
        .estop(estop_at_sample), .hb(hb_pend), .fclr(fclr_pend), .lat_fault(latf_pend), .adc_fault(adc_fault),
        .ctrl(ctrl),
        .bq_b0(mem[A_BQ_B0]), .bq_b1(mem[A_BQ_B1]), .bq_b2(mem[A_BQ_B2]), .bq_a1(mem[A_BQ_A1]), .bq_a2(mem[A_BQ_A2]),
        .filt_bypass(mem[A_FILT_BYPASS]), .fs_hz(mem[A_FS_HZ]), .enc_gain(mem[A_ENC_GAIN]),
        .adc_fs({mem[A_ADC_FS7], mem[A_ADC_FS6], mem[A_ADC_FS5], mem[A_ADC_FS4],
                 mem[A_ADC_FS3], mem[A_ADC_FS2], mem[A_ADC_FS1], mem[A_ADC_FS0]}),
        .f_offset(mem[A_F_OFFSET]), .v_alpha(mem[A_V_ALPHA]),
        .adm_cb(mem[A_ADM_CB]), .adm_cf(mem[A_ADM_CF]), .adm_ck(mem[A_ADM_CK]), .x_eq(mem[A_X_EQ]), .c_dt(mem[A_C_DT]),
        .kp(mem[A_KP]), .kd(mem[A_KD]), .c_duty(mem[A_C_DUTY]), .slew(mem[A_SLEW]),
        .pc_alpha_max(mem[A_PC_ALPHA_MAX]), .pc_v_eps(mem[A_PC_V_EPS]),
        .lim_f(mem[A_LIM_F]), .lim_v(mem[A_LIM_V]), .lim_xmin(mem[A_LIM_XMIN]), .lim_xmax(mem[A_LIM_XMAX]),
        .wdog_ticks(mem[A_WDOG_TICKS]),
        .lif_leak(mem[A_LIF_LEAK]),
        .lif_gain({mem[A_LIF_GAIN3], mem[A_LIF_GAIN2], mem[A_LIF_GAIN1], mem[A_LIF_GAIN0]}),
        .lif_theta(mem[A_LIF_THETA]),
        .gate_isi(mem[A_GATE_ISI]), .gate_ratio(mem[A_GATE_RATIO]), .stim_amp(mem[A_STIM_AMP]),
        .rep_lo(mem[A_REP_LO]), .rep_hi(mem[A_REP_HI]),
        .done(core_done), .tick_tlm(tick_tlm), .F(F), .x(x), .v(v), .x_cmd(x_cmd), .v_cmd(v_cmd), .e(e), .duty(duty),
        .filt_flat(filt_flat), .spikes(spikes), .gated(gated), .stim_amp_out(stim_amp_k), .status(status),
        .rep_count(rep_count), .spike_count(spike_count), .pwm_cmp(pwm_cmp), .pwm_rev(pwm_rev),
        .active_o(core_active), .fault_o(core_fault), .busy(core_busy)
    );

    // one-shot host flags are consumed by the tick that starts after they were written
    always @(posedge clk) begin
        if (rst) begin
            fclr_pend <= 1'b0; hb_pend <= 1'b0; latf_pend <= 1'b0;
        end else begin
            if (adc_done) begin fclr_pend <= 1'b0; hb_pend <= 1'b0; latf_pend <= 1'b0; end
            if (csr_wr && csr_addr == A_FAULT_CLEAR && csr_wdata[0]) fclr_pend <= 1'b1;
            if (csr_wr && csr_addr == A_HEARTBEAT) hb_pend <= 1'b1;
            if (lat_overrun) latf_pend <= 1'b1;
        end
    end

    // ------------------------------------------------------------------ actuation
    wire pwm_loaded;
    wire drive_en, stim_en;
    apt_interlock u_ilk (.core_active(core_active), .core_fault(core_fault), .estop(estop_now),
                         .stim_ctrl(ctrl[2]), .drive_en(drive_en), .stim_en(stim_en));
    apt_pwm #(.PERIOD(PWM_PERIOD), .DEADTIME(PWM_DEADTIME)) u_pwm (
        .clk(clk), .rst(rst), .enable(drive_en), .update(core_done), .cmp_in(pwm_cmp), .rev_in(pwm_rev),
        .loaded(pwm_loaded), .ha(pwm_ha), .la(pwm_la), .hb(pwm_hb), .lb(pwm_lb)
    );

    wire [3:0] onset, overrun;
    genvar k;
    generate
        for (k = 0; k < 4; k = k + 1) begin : g_stim
            apt_stim u_stim (
                .clk(clk), .rst(rst), .enable(stim_en), .trigger(core_done && spikes[k]),
                .amp(stim_amp_k[8*k +: 8]), .width(mem[A_STIM_WIDTH]), .gap(mem[A_STIM_GAP]),
                .refract(mem[A_STIM_REFRACT]), .cath(stim_cath[k]), .anod(stim_anod[k]),
                .dac(stim_dac[8*k +: 8]), .onset(onset[k]), .overrun(overrun[k])
            );
        end
    endgenerate

    reg [31:0] stim_overruns;
    always @(posedge clk)
        if (rst) stim_overruns <= 0;
        else stim_overruns <= stim_overruns + overrun[0] + overrun[1] + overrun[2] + overrun[3];

    // ------------------------------------------------------------------ latency monitor
    wire [31:0] lat_m_last, lat_m_max, lat_a_last, lat_a_max;
    apt_latency u_lat (
        .clk(clk), .rst(rst), .sample(tick), .core_done(core_done), .any_spike(|spikes && stim_en),
        .pwm_loaded(pwm_loaded), .stim_onset(|onset), .budget(mem[A_LAT_BUDGET]),
        .motor_last(lat_m_last), .motor_max(lat_m_max), .aff_last(lat_a_last), .aff_max(lat_a_max),
        .overrun(lat_overrun)
    );

    function [15:0] sat16;
        input [31:0] a;
        begin sat16 = (a > 32'h0000FFFF) ? 16'hFFFF : a[15:0]; end
    endfunction

    // ------------------------------------------------------------------ telemetry
    wire [335:0] payload = {sat16(lat_a_last), sat16(lat_m_last), rep_count, gated, spikes,
                            duty, e, v_cmd, x_cmd, v, x, F, status, tick_tlm, 8'hA1};
    wire [31:0] tlm_dropped;
    wire        tlm_busy;
    apt_telemetry #(.UART_DIV(UART_DIVP)) u_tlm (
        .clk(clk), .rst(rst), .send(core_done && ctrl[3]), .payload(payload), .tx(tlm_tx),
        .dropped(tlm_dropped), .busy(tlm_busy)
    );

    // ------------------------------------------------------------------ CSR access
    always @(posedge clk) begin
        if (rst) csr_reset;
        else if (csr_wr && is_rw(csr_addr)) mem[csr_addr] <= csr_wdata;
    end

    always @(posedge clk) begin
        if (csr_rd) begin
            case (csr_addr)
                A_ID:             csr_rdata <= DEVICE_ID;
                A_VERSION:        csr_rdata <= RTL_VERSION;
                A_STATUS:         csr_rdata <= {16'd0, status};
                A_TICK:           csr_rdata <= tick_tlm;
                A_X_MEAS:         csr_rdata <= x;
                A_V_MEAS:         csr_rdata <= v;
                A_F_MEAS:         csr_rdata <= F;
                A_E_OBS:          csr_rdata <= e;
                A_DUTY:           csr_rdata <= duty;
                A_X_CMD:          csr_rdata <= x_cmd;
                A_LAT_MOTOR_LAST: csr_rdata <= lat_m_last;
                A_LAT_MOTOR_MAX:  csr_rdata <= lat_m_max;
                A_LAT_AFF_LAST:   csr_rdata <= lat_a_last;
                A_LAT_AFF_MAX:    csr_rdata <= lat_a_max;
                A_REP_COUNT:      csr_rdata <= {16'd0, rep_count};
                A_SPIKE_COUNT:    csr_rdata <= spike_count;
                A_STIM_OVERRUN:   csr_rdata <= stim_overruns;
                A_ENC_ERR:        csr_rdata <= enc_errors;
                A_FILT0, A_FILT1, A_FILT2, A_FILT3, A_FILT4, A_FILT5, A_FILT6, A_FILT7:
                                  csr_rdata <= filt_flat[32*(csr_addr - A_FILT0) +: 32];
                default:          csr_rdata <= is_rw(csr_addr) ? mem[csr_addr] : 32'd0;
            endcase
        end
    end

    assign fault_led = core_fault;


endmodule

`default_nettype wire
