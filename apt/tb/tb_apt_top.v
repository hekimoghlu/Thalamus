// tb_apt_top.v - bit-exact co-simulation of the FPGA against the C++ golden model.
//
// `aptctl vectors` runs the golden model closed-loop on the HIL plant and writes:
//   cfg.txt     CSR image (addr value), CTRL with RUN last
//   stim.txt    per tick: 8 ADC codes, encoder count, flags (b0 e-stop, b1 fault-clear)
//   expect.txt  per tick: the 42-byte telemetry payload the golden model emitted
// This bench replays that sensor stream through behavioural models of the ADC and encoder,
// configures the device over SPI exactly as the MCU would, decodes the UART/SLIP telemetry,
// checks every CRC, and compares every payload byte except the two cycle-latency fields.
// Alongside, it checks invariants the golden model does not cover: H-bridge dead-time,
// stimulator charge balance, e-stop gate cut-off time, and the host watchdog.
`timescale 1ns / 1ps
`default_nettype none

module tb_apt_top;
    localparam integer TICK_DIV = 5000;   // simulation-scale tick (the model is rate-free)
    localparam integer UDIV     = 4;      // simulation-scale UART
    localparam integer MAXT     = 4096;

    reg clk = 0;
    always #10 clk = ~clk;  // 50 MHz
    reg rst_n = 0;

    // ---------------------------------------------------------------- DUT
    wire adc_convst, adc_cs_n, adc_sclk;
    reg  adc_busy = 0, adc_dout = 0;
    reg  enc_a = 0, enc_b = 0;
    wire pwm_ha, pwm_la, pwm_hb, pwm_lb;
    wire [3:0] stim_cath, stim_anod;
    wire [31:0] stim_dac;
    reg  estop_n = 1;
    wire fault_led, sample_o;
    reg  spi_sclk = 0, spi_cs_n = 1, spi_mosi = 0;
    wire spi_miso, tlm_tx;

    apt_top #(.TICK_DIV(TICK_DIV), .UART_DIVP(UDIV)) dut (
        .clk(clk), .rst_n(rst_n),
        .adc_convst(adc_convst), .adc_busy(adc_busy), .adc_cs_n(adc_cs_n), .adc_sclk(adc_sclk), .adc_dout(adc_dout),
        .enc_a(enc_a), .enc_b(enc_b),
        .pwm_ha(pwm_ha), .pwm_la(pwm_la), .pwm_hb(pwm_hb), .pwm_lb(pwm_lb),
        .stim_cath(stim_cath), .stim_anod(stim_anod), .stim_dac(stim_dac),
        .estop_n(estop_n), .fault_led(fault_led), .sample_o(sample_o),
        .spi_sclk(spi_sclk), .spi_cs_n(spi_cs_n), .spi_mosi(spi_mosi), .spi_miso(spi_miso),
        .tlm_tx(tlm_tx)
    );

    // ---------------------------------------------------------------- vectors
    reg [15:0]  codes [0:MAXT-1][0:7];
    reg [31:0]  encv  [0:MAXT-1];
    reg [7:0]   flags [0:MAXT-1];
    reg [335:0] expv  [0:MAXT-1];
    reg [6:0]   cfg_a [0:255];
    reg [31:0]  cfg_d [0:255];
    integer     nticks, ncfg;

    initial begin : load
        integer fd, r, t;
        reg [31:0] c0, c1, c2, c3, c4, c5, c6, c7, en, fl, a, d;
        fd = $fopen("tb/vectors/stim.txt", "r");   // run from apt/ (scripts/sim.sh)
        if (fd == 0) begin $display("FATAL: cannot open tb/vectors/stim.txt - run `aptctl vectors`"); $finish; end
        nticks = 0;
        while (!$feof(fd) && nticks < MAXT) begin
            r = $fscanf(fd, "%h %h %h %h %h %h %h %h %h %h\n", c0, c1, c2, c3, c4, c5, c6, c7, en, fl);
            if (r == 10) begin
                codes[nticks][0] = c0; codes[nticks][1] = c1; codes[nticks][2] = c2; codes[nticks][3] = c3;
                codes[nticks][4] = c4; codes[nticks][5] = c5; codes[nticks][6] = c6; codes[nticks][7] = c7;
                encv[nticks] = en; flags[nticks] = fl[7:0];
                nticks = nticks + 1;
            end
        end
        $fclose(fd);
        $readmemh("tb/vectors/expect.txt", expv, 0, nticks - 1);
        fd = $fopen("tb/vectors/cfg.txt", "r");
        ncfg = 0;
        while (!$feof(fd)) begin
            r = $fscanf(fd, "%h %h\n", a, d);
            if (r == 2) begin cfg_a[ncfg] = a[6:0]; cfg_d[ncfg] = d; ncfg = ncfg + 1; end
        end
        $fclose(fd);
        $display("loaded %0d ticks, %0d CSR writes", nticks, ncfg);
    end

    // ---------------------------------------------------------------- ADC model
    integer conv_idx = 0;
    reg [127:0] adc_sh;
    always @(posedge adc_convst) begin : conv
        integer c;
        // after the replay, hold the last sample (the directed tests run on a quiet plant)
        for (c = 0; c < 8; c = c + 1) adc_sh[127 - 16*c -: 16] = codes[(conv_idx < nticks) ? conv_idx : nticks - 1][c];
        conv_idx = conv_idx + 1;
        adc_busy <= 1;
        repeat (100) @(posedge clk);
        adc_busy <= 0;
    end
    always @(negedge adc_cs_n) adc_dout <= adc_sh[127];
    always @(negedge adc_sclk) if (!adc_cs_n) begin
        adc_sh = {adc_sh[126:0], 1'b0};
        adc_dout <= adc_sh[127];
    end

    // ---------------------------------------------------------------- encoder model
    integer enc_cur = 0;
    task enc_step(input integer dir);
        begin
            enc_cur = enc_cur + dir;
            case (enc_cur & 3)
                0: {enc_a, enc_b} = 2'b00;
                1: {enc_a, enc_b} = 2'b01;
                2: {enc_a, enc_b} = 2'b11;
                3: {enc_a, enc_b} = 2'b10;
            endcase
            repeat (8) @(posedge clk);
        end
    endtask
    task enc_goto(input integer target);
        begin
            while (enc_cur < target) enc_step(1);
            while (enc_cur > target) enc_step(-1);
        end
    endtask

    // ---------------------------------------------------------------- SPI host (MCU side)
    task spi_xfer(input rw, input [6:0] addr, input [31:0] wdata, output [31:0] rdata);
        integer b;
        reg [39:0] frame;
        begin
            frame = {rw, addr, wdata};
            rdata = 0;
            spi_cs_n = 0;
            repeat (8) @(posedge clk);
            for (b = 39; b >= 0; b = b - 1) begin
                spi_mosi = frame[b];
                repeat (8) @(posedge clk);
                spi_sclk = 1;
                if (b < 32) rdata = {rdata[30:0], spi_miso};
                repeat (8) @(posedge clk);
                spi_sclk = 0;
            end
            repeat (8) @(posedge clk);
            spi_cs_n = 1;
            repeat (16) @(posedge clk);
        end
    endtask
    task csr_write(input [6:0] a, input [31:0] d);
        reg [31:0] dummy;
        begin spi_xfer(1'b1, a, d, dummy); end
    endtask
    task csr_read(input [6:0] a, output [31:0] d);
        begin spi_xfer(1'b0, a, 32'd0, d); end
    endtask

    // ---------------------------------------------------------------- UART + SLIP receiver
    integer frames = 0, mismatches = 0, crc_errors = 0, len_errors = 0, compared = 0;
    integer tlm_lat_m_max = 0, tlm_lat_a_max = 0, tlm_lat_a_tick = 0;
    reg [7:0] fb [0:63];
    integer flen = 0;
    reg esc = 0;

    function [15:0] crc_byte(input [15:0] c, input [7:0] b);
        integer k;
        reg [15:0] r;
        begin
            r = c ^ {b, 8'h00};
            for (k = 0; k < 8; k = k + 1) r = r[15] ? ((r << 1) ^ 16'h1021) : (r << 1);
            crc_byte = r;
        end
    endfunction

    task check_frame;
        integer i, t;
        reg [15:0] crc;
        reg [7:0] eb;
        begin
            frames = frames + 1;
            if (flen != 44) begin
                len_errors = len_errors + 1;
                $display("  frame %0d: length %0d", frames, flen);
            end else begin
                crc = 16'hFFFF;
                for (i = 0; i < 42; i = i + 1) crc = crc_byte(crc, fb[i]);
                if (crc != {fb[43], fb[42]}) begin
                    crc_errors = crc_errors + 1;
                    $display("  frame %0d: CRC %h != %h", frames, {fb[43], fb[42]}, crc);
                end
                t = {fb[4], fb[3], fb[2], fb[1]};
                if ({fb[39], fb[38]} > tlm_lat_m_max) tlm_lat_m_max = {fb[39], fb[38]};
                if ({fb[41], fb[40]} > tlm_lat_a_max) begin tlm_lat_a_max = {fb[41], fb[40]}; tlm_lat_a_tick = t; end
                if (t < nticks) begin
                    compared = compared + 1;
                    for (i = 0; i < 42; i = i + 1) begin
                        eb = expv[t][335 - 8*i -: 8];
                        if (!(i >= 38 && i <= 41) && fb[i] !== eb) begin
                            mismatches = mismatches + 1;
                            if (mismatches <= 12)
                                $display("  MISMATCH tick %0d byte %0d: rtl %h model %h", t, i, fb[i], eb);
                        end
                    end
                end
            end
        end
    endtask

    initial begin : uart_rx
        integer i;
        reg [7:0] b;
        forever begin
            @(negedge tlm_tx);
            repeat (UDIV / 2) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin
                repeat (UDIV) @(posedge clk);
                b[i] = tlm_tx;
            end
            repeat (UDIV) @(posedge clk);  // stop bit
            if (b == 8'hC0) begin
                if (flen > 0) check_frame;
                flen = 0; esc = 0;
            end else if (esc) begin
                fb[flen] = (b == 8'hDC) ? 8'hC0 : (b == 8'hDD) ? 8'hDB : b;
                flen = flen + 1; esc = 0;
            end else if (b == 8'hDB) esc = 1;
            else begin fb[flen] = b; flen = flen + 1; end
        end
    end

    // ---------------------------------------------------------------- invariants
    integer deadtime_violations = 0;
    integer cath_cycles [0:3];
    integer anod_cycles [0:3];
    integer pulses = 0;
    initial begin : inv_init
        integer k;
        for (k = 0; k < 4; k = k + 1) begin cath_cycles[k] = 0; anod_cycles[k] = 0; end
    end
    always @(posedge clk) begin : invariants
        integer k;
        if ((pwm_ha && pwm_la) || (pwm_hb && pwm_lb)) deadtime_violations = deadtime_violations + 1;
        for (k = 0; k < 4; k = k + 1) begin
            if (stim_cath[k]) cath_cycles[k] = cath_cycles[k] + 1;
            if (stim_anod[k]) anod_cycles[k] = anod_cycles[k] + 1;
            if (stim_cath[k] && stim_anod[k]) deadtime_violations = deadtime_violations + 1;
        end
    end
    always @(posedge stim_cath[0] or posedge stim_cath[1] or posedge stim_cath[2] or posedge stim_cath[3])
        pulses = pulses + 1;

    // ---------------------------------------------------------------- stimulus sequencing
    integer failures = 0;
    initial begin : main
        integer i, t;
        reg [31:0] rd;
        reg ok;
        wait (nticks > 0);
        // position the encoder at tick 0 before reset releases the counter
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (20) @(posedge clk);
        enc_goto($signed(encv[0]));

        csr_read(7'h00, rd);
        if (rd !== 32'h41505431) begin $display("FAIL: ID = %h", rd); failures = failures + 1; end
        else $display("device ID %h ('APT1')", rd);
        for (i = 0; i < ncfg; i = i + 1) csr_write(cfg_a[i], cfg_d[i]);
        csr_read(cfg_a[0], rd);
        if (rd !== cfg_d[0]) begin $display("FAIL: CSR readback %h != %h", rd, cfg_d[0]); failures = failures + 1; end

        for (t = 0; t < nticks; t = t + 1) begin
            @(posedge dut.core_done);     // tick t computed: stage tick t+1
            if (t + 1 < nticks) begin
                estop_n = !flags[t + 1][0];
                if (flags[t + 1][1]) csr_write(7'h04, 32'd1);   // FAULT_CLEAR
                enc_goto($signed(encv[t + 1]));
            end
            if (t % 200 == 0) $display("  tick %0d / %0d  (frames %0d, mismatches %0d)", t, nticks, frames, mismatches);
        end
        repeat (4 * TICK_DIV) @(posedge clk);

        $display("");
        $display("telemetry: %0d frames, %0d compared against the golden model, %0d byte mismatches, %0d CRC errors, %0d length errors",
                 frames, compared, mismatches, crc_errors, len_errors);
        if (compared < nticks - 2 || mismatches != 0 || crc_errors != 0 || len_errors != 0) failures = failures + 1;

        $display("telemetry latency fields: motor max %0d cycles, afferent max %0d cycles (frame %0d)",
                 tlm_lat_m_max, tlm_lat_a_max, tlm_lat_a_tick);
        csr_read(7'h49, rd);
        $display("latency monitor: sample->PWM max %0d cycles (%0d ns)", rd, rd * 20);
        csr_read(7'h4B, rd);
        $display("latency monitor: sample->stimulus max %0d cycles (%0d ns)", rd, rd * 20);
        csr_read(7'h4D, rd);
        $display("afferent events: %0d, stimulator pulses started: %0d", rd, pulses);

        // charge balance: every pulse that started has finished with equal phases
        ok = 1;
        for (i = 0; i < 4; i = i + 1) if (cath_cycles[i] != anod_cycles[i]) ok = 0;
        $display("charge balance: cathodic/anodic cycles ch0 %0d/%0d ch1 %0d/%0d ch2 %0d/%0d ch3 %0d/%0d -> %0s",
                 cath_cycles[0], anod_cycles[0], cath_cycles[1], anod_cycles[1],
                 cath_cycles[2], anod_cycles[2], cath_cycles[3], anod_cycles[3], ok ? "balanced" : "UNBALANCED");
        if (!ok) failures = failures + 1;
        $display("dead-time / shoot-through violations: %0d", deadtime_violations);
        if (deadtime_violations != 0) failures = failures + 1;

        // e-stop: gate drives must drop within a few clocks, before the next tick
        begin : estop_test
            integer n;
            wait (pwm_ha || pwm_hb || pwm_la || pwm_lb);
            estop_n = 0;
            n = 0;
            while ((pwm_ha || pwm_hb || pwm_la || pwm_lb) && n < 100) begin @(posedge clk); n = n + 1; end
            $display("e-stop: all gates off %0d cycles after the pin (%0d ns)", n, n * 20);
            if (n > 5) failures = failures + 1;
            estop_n = 1;
        end

        // watchdog: arm a 5-tick timeout, stop the heartbeat, expect a latched WATCHDOG cause
        csr_write(7'h06, 32'd5);
        csr_write(7'h04, 32'd1);
        repeat (12) @(posedge dut.core_done);
        csr_read(7'h03, rd);
        $display("watchdog: STATUS %h -> %0s", rd, (rd[1] && rd[12]) ? "tripped as specified" : "DID NOT TRIP");
        if (!(rd[1] && rd[12])) failures = failures + 1;

        $display("");
        if (failures == 0) $display("PASS: RTL is bit-exact with the golden model and all invariants hold");
        else               $display("FAIL: %0d check(s) failed", failures);
        $finish;
    end

    initial begin
        #(64'd400_000_000);
        $display("FAIL: timeout");
        $finish;
    end
endmodule

`default_nettype wire
