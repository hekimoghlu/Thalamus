// apt_core.v - the per-tick datapath of the Artificial Peripheral Thalamus.
//
// Bit-exact implementation of include/apt/core_model.hpp. One 32x32 multiplier and one 64-bit
// accumulator are time-shared by an explicit state machine; every multiply-accumulate of the
// golden model is one state here, in the same order, with the same rounding (add half an LSB,
// arithmetic shift, saturate to 32 bits). A tick takes ~190 cycles (+64 when the passivity
// controller divides) - under 6 us at 50 MHz against a 1 ms control period.
//
// Formats: signals Q12.20, coefficients Q8.24, ADC codes Q1.15 of full scale.
`default_nettype none

module apt_core #(
    parameter integer PWM_PERIOD = 1250
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 run,        // low: datapath held in its reset state
    input  wire                 start,      // inputs below are valid for this tick
    input  wire [16*8-1:0]      codes,      // ch7..ch0, 16-bit two's complement
    input  wire signed [31:0]   enc,
    input  wire                 estop,
    input  wire                 hb,
    input  wire                 fclr,
    input  wire                 lat_fault,
    input  wire                 adc_fault,
    // configuration (CSRs)
    input  wire [31:0]          ctrl,
    input  wire signed [31:0]   bq_b0, bq_b1, bq_b2, bq_a1, bq_a2,
    input  wire [31:0]          filt_bypass,
    input  wire [31:0]          fs_hz,
    input  wire signed [31:0]   enc_gain,
    input  wire [32*8-1:0]      adc_fs,
    input  wire signed [31:0]   f_offset, v_alpha,
    input  wire signed [31:0]   adm_cb, adm_cf, adm_ck, x_eq, c_dt,
    input  wire signed [31:0]   kp, kd, c_duty, slew,
    input  wire signed [31:0]   pc_alpha_max, pc_v_eps,
    input  wire signed [31:0]   lim_f, lim_v, lim_xmin, lim_xmax,
    input  wire [31:0]          wdog_ticks,
    input  wire signed [31:0]   lif_leak,
    input  wire [32*4-1:0]      lif_gain,
    input  wire signed [31:0]   lif_theta,
    input  wire [31:0]          gate_isi, gate_ratio, stim_amp,
    input  wire signed [31:0]   rep_lo, rep_hi,
    // results (valid from `done` until the next `start`)
    output reg                  done,
    output reg  [31:0]          tick_tlm,
    output reg  signed [31:0]   F, x, v, x_cmd, v_cmd, e, duty,
    output reg  [32*8-1:0]      filt_flat,
    output reg  [3:0]           spikes,
    output reg  [3:0]           gated,
    output reg  [31:0]          stim_amp_out,
    output reg  [15:0]          status,
    output reg  [15:0]          rep_count,
    output reg  [31:0]          spike_count,
    output reg  [31:0]          pwm_cmp,
    output reg                  pwm_rev,
    output reg                  active_o,
    output reg                  fault_o,
    output wire                 busy
);

    // ------------------------------------------------------------------ fixed-point helpers
    function signed [31:0] sat64;
        input signed [63:0] a;
        begin
            if (a > 64'sh000000007FFFFFFF)      sat64 = 32'h7FFFFFFF;
            else if (a < 64'shFFFFFFFF80000000) sat64 = 32'h80000000;
            else                                sat64 = a[31:0];
        end
    endfunction

    function signed [31:0] rnd24;  // signal x coef
        input signed [63:0] a;
        reg signed [63:0] t;
        begin t = (a + 64'sd8388608) >>> 24; rnd24 = sat64(t); end
    endfunction

    function signed [31:0] rnd20;  // signal x signal
        input signed [63:0] a;
        reg signed [63:0] t;
        begin t = (a + 64'sd524288) >>> 20; rnd20 = sat64(t); end
    endfunction

    function signed [31:0] rnd15;  // code x full scale
        input signed [63:0] a;
        reg signed [63:0] t;
        begin t = (a + 64'sd16384) >>> 15; rnd15 = sat64(t); end
    endfunction

    function signed [31:0] rnd4;   // integer x coef
        input signed [63:0] a;
        reg signed [63:0] t;
        begin t = (a + 64'sd8) >>> 4; rnd4 = sat64(t); end
    endfunction

    function signed [31:0] sadd;
        input signed [31:0] a, b;
        reg signed [63:0] t;
        begin t = a; t = t + b; sadd = sat64(t); end
    endfunction

    function signed [31:0] ssub;
        input signed [31:0] a, b;
        reg signed [63:0] t;
        begin t = a; t = t - b; ssub = sat64(t); end
    endfunction

    function signed [31:0] sneg;
        input signed [31:0] a;
        reg signed [63:0] t;
        begin t = a; t = -t; sneg = sat64(t); end
    endfunction

    function signed [31:0] sabs;
        input signed [31:0] a;
        reg signed [63:0] t;
        begin t = a; if (t < 0) t = -t; sabs = sat64(t); end
    endfunction

    function signed [31:0] sclamp;
        input signed [31:0] a, lo, hi;
        begin sclamp = (a < lo) ? lo : (a > hi) ? hi : a; end
    endfunction

    function [2:0] stim_src;  // afferent channel k <- ADC channel {1, 0, 3, 4}[k]
        input [1:0] k;
        begin
            case (k)
                2'd0: stim_src = 3'd1;   // grip force
                2'd1: stim_src = 3'd0;   // interaction force
                2'd2: stim_src = 3'd3;   // MCP angle
                default: stim_src = 3'd4; // PIP angle
            endcase
        end
    endfunction

    localparam signed [31:0] ONE = 32'sh00100000;  // 1.0 in Q12.20

    // ------------------------------------------------------------------ state
    localparam [5:0]
        S_IDLE = 6'd0,  S_SC  = 6'd1,  S_SCR = 6'd2,
        S_F0   = 6'd3,  S_F1  = 6'd4,  S_F2  = 6'd5,  S_F3  = 6'd6,  S_F4  = 6'd7,  S_F5 = 6'd8,
        S_X0   = 6'd9,  S_X1  = 6'd10, S_V0  = 6'd11, S_V1  = 6'd12, S_V2  = 6'd13, S_V3 = 6'd14,
        S_SAFE = 6'd15, S_P0  = 6'd16, S_P1  = 6'd17, S_P2  = 6'd18, S_P3  = 6'd19,
        S_PC0  = 6'd20, S_PC1 = 6'd21, S_PC2 = 6'd22, S_PC3 = 6'd23, S_PC4 = 6'd24, S_PC5 = 6'd25,
        S_PC6  = 6'd26, S_PC7 = 6'd27, S_PC8 = 6'd28, S_PC9 = 6'd29, S_PC10 = 6'd30, S_PC11 = 6'd31,
        S_A0   = 6'd32, S_A1  = 6'd33, S_A2  = 6'd34, S_A3  = 6'd35, S_A4  = 6'd36, S_A5 = 6'd37,
        S_I0   = 6'd38, S_I1  = 6'd39, S_I2  = 6'd40, S_I3  = 6'd41, S_I4  = 6'd42,
        S_L0   = 6'd43, S_L1  = 6'd44, S_L2  = 6'd45, S_L3  = 6'd46,
        S_REP  = 6'd47, S_D0  = 6'd48, S_D1  = 6'd49;

    reg [5:0] st;
    reg [2:0] ch;
    reg [1:0] kk;
    assign busy = (st != S_IDLE);

    // latched tick inputs
    reg [16*8-1:0]    codes_r;
    reg signed [31:0] enc_r;
    reg               estop_r, hb_r, fclr_r, latf_r, adcf_r;

    // per-channel filter history and outputs
    reg signed [31:0] x1 [0:7];
    reg signed [31:0] x2 [0:7];
    reg signed [31:0] y1 [0:7];
    reg signed [31:0] y2 [0:7];
    reg signed [31:0] filt [0:7];
    // afferent relay state
    reg signed [31:0] lif_v [0:3];
    reg [15:0]        since [0:3];

    reg signed [31:0] phys, x_prev, v_raw, p, v2, den, alpha, f_pc, tmp, lt, err, fin, u;
    reg               primed, armed, pc_active, rep_state;
    reg [7:0]         latched;
    reg [31:0]        wdog, tick_cnt;

    // ------------------------------------------------------------------ shared MAC
    localparam [1:0] ACC_HOLD = 2'd0, ACC_LOAD = 2'd1, ACC_ADD = 2'd2, ACC_SUB = 2'd3;
    reg signed [31:0] ma, mb;
    reg [1:0]         aop;
    reg signed [63:0] acc;
    wire signed [63:0] prod = ma * mb;

    wire [2:0] src_ch = stim_src(kk);

    always @* begin
        ma = 32'sd0;
        mb = 32'sd0;
        aop = ACC_HOLD;
        case (st)
            S_SC:  begin ma = {{16{codes_r[16*ch+15]}}, codes_r[16*ch +: 16]}; mb = adc_fs[32*ch +: 32]; aop = ACC_LOAD; end
            S_F0:  begin ma = phys;   mb = bq_b0; aop = ACC_LOAD; end
            S_F1:  begin ma = x1[ch]; mb = bq_b1; aop = ACC_ADD;  end
            S_F2:  begin ma = x2[ch]; mb = bq_b2; aop = ACC_ADD;  end
            S_F3:  begin ma = y1[ch]; mb = bq_a1; aop = ACC_SUB;  end
            S_F4:  begin ma = y2[ch]; mb = bq_a2; aop = ACC_SUB;  end
            S_X0:  begin ma = enc_r;  mb = enc_gain; aop = ACC_LOAD; end
            S_V0:  begin ma = ssub(x, x_prev); mb = {16'd0, fs_hz[15:0]}; aop = ACC_LOAD; end
            S_V2:  begin ma = ssub(v_raw, v);  mb = v_alpha; aop = ACC_LOAD; end
            S_P0:  begin ma = F;      mb = v;     aop = ACC_LOAD; end
            S_P2:  begin ma = p;      mb = c_dt;  aop = ACC_LOAD; end
            S_PC0: begin ma = v;      mb = v;     aop = ACC_LOAD; end
            S_PC2: begin ma = v2;     mb = c_dt;  aop = ACC_LOAD; end
            S_PC6: begin ma = alpha;  mb = v;     aop = ACC_LOAD; end
            S_PC8: begin ma = alpha;  mb = v2;    aop = ACC_LOAD; end
            S_PC10:begin ma = tmp;    mb = c_dt;  aop = ACC_LOAD; end
            S_A0:  begin ma = v_cmd;  mb = adm_cb; aop = ACC_LOAD; end
            S_A1:  begin ma = fin;    mb = adm_cf; aop = ACC_ADD;  end
            S_A2:  begin ma = err;    mb = adm_ck; aop = ACC_SUB;  end
            S_A4:  begin ma = v_cmd;  mb = c_dt;   aop = ACC_LOAD; end
            S_I0:  begin ma = ssub(x_cmd, x); mb = kp; aop = ACC_LOAD; end
            S_I1:  begin ma = ssub(v_cmd, v); mb = kd; aop = ACC_ADD;  end
            S_I3:  begin ma = u;      mb = c_duty; aop = ACC_LOAD; end
            S_L0:  begin ma = lif_v[kk]; mb = lif_leak; aop = ACC_LOAD; end
            S_L2:  begin ma = sabs(filt[src_ch]); mb = lif_gain[32*kk +: 32]; aop = ACC_LOAD; end
            S_D0:  begin ma = sabs(duty); mb = PWM_PERIOD; aop = ACC_LOAD; end
            default: ;
        endcase
    end

    always @(posedge clk) begin
        case (aop)
            ACC_LOAD: acc <= prod;
            ACC_ADD:  acc <= acc + prod;
            ACC_SUB:  acc <= acc - prod;
            default:  ;
        endcase
    end

    // ------------------------------------------------------------------ divider (passivity ctrl)
    reg         div_start;
    wire        div_done;
    wire [63:0] div_q;
    reg  [63:0] div_n;
    apt_udiv u_div (
        .clk(clk), .rst(rst), .start(div_start), .dividend(div_n), .divisor(den[31:0]),
        .busy(), .done(div_done), .quotient(div_q)
    );

    // ------------------------------------------------------------------ safety combinational
    wire        en_bit   = ctrl[0];
    wire        pc_bit   = ctrl[1];
    wire        stim_bit = ctrl[2];
    reg  [31:0] wdog_n;
    reg  [7:0]  cause, latched_n;
    always @* begin
        cause = 8'd0;
        if (estop_r)                         cause = cause | 8'h01;
        if (sabs(F) > lim_f)                 cause = cause | 8'h02;
        if (sabs(v) > lim_v)                 cause = cause | 8'h04;
        if (x < lim_xmin || x > lim_xmax)    cause = cause | 8'h08;
        wdog_n = 32'd0;
        if (wdog_ticks != 32'd0) begin
            wdog_n = hb_r ? 32'd0 : (wdog == 32'hFFFFFFFF ? wdog : wdog + 32'd1);
            if (wdog_n >= wdog_ticks)        cause = cause | 8'h10;
        end
        if (latf_r)                          cause = cause | 8'h20;
        if (adcf_r)                          cause = cause | 8'h40;
        latched_n = fclr_r ? cause : (latched | cause);
    end

    // LIF step result (state S_L3)
    wire signed [31:0] lif_vn    = sadd(lt, rnd24(acc));
    wire               lif_spike = (lif_theta > 0) && (lif_vn >= lif_theta);
    wire [7:0]         amp_base  = stim_amp[8*kk +: 8];
    wire [39:0]        amp_prod  = amp_base * gate_ratio;
    wire               in_gate   = {16'd0, since[kk]} < gate_isi;

    // duty update (state S_I4)
    wire signed [31:0] d_new   = sclamp(rnd24(acc), -ONE, ONE);
    wire signed [31:0] d_delta = sclamp(ssub(d_new, duty), sneg(slew), slew);

    integer i;
    always @(posedge clk) begin
        done      <= 1'b0;
        div_start <= 1'b0;
        if (rst || !run) begin
            st <= S_IDLE;
            ch <= 3'd0;
            kk <= 2'd0;
            for (i = 0; i < 8; i = i + 1) begin
                x1[i] <= 0; x2[i] <= 0; y1[i] <= 0; y2[i] <= 0; filt[i] <= 0;
            end
            for (i = 0; i < 4; i = i + 1) begin
                lif_v[i] <= 0; since[i] <= 16'hFFFF;
            end
            x_prev <= 0; v <= 0; e <= 0; x_cmd <= 0; v_cmd <= 0; duty <= 0; F <= 0; x <= 0;
            f_pc <= 0; pc_active <= 1'b0; primed <= 1'b0; armed <= 1'b0; rep_state <= 1'b0;
            latched <= 8'd0; wdog <= 32'd0; tick_cnt <= 32'd0; tick_tlm <= 32'd0;
            rep_count <= 16'd0; spike_count <= 32'd0; spikes <= 4'd0; gated <= 4'd0;
            stim_amp_out <= 32'd0; status <= 16'd0; pwm_cmp <= 32'd0; pwm_rev <= 1'b0;
            active_o <= 1'b0; fault_o <= 1'b0; filt_flat <= 0;
        end else begin
            case (st)
                S_IDLE: if (start) begin
                    codes_r <= codes; enc_r <= enc;
                    estop_r <= estop; hb_r <= hb; fclr_r <= fclr; latf_r <= lat_fault; adcf_r <= adc_fault;
                    spikes <= 4'd0; gated <= 4'd0; stim_amp_out <= 32'd0;
                    ch <= 3'd0;
                    st <= S_SC;
                end
                // 1-2  scale + DF-I biquad, per channel
                S_SC:  st <= S_SCR;
                S_SCR: begin phys <= rnd15(acc); st <= S_F0; end
                S_F0:  st <= S_F1;
                S_F1:  st <= S_F2;
                S_F2:  st <= S_F3;
                S_F3:  st <= S_F4;
                S_F4:  st <= S_F5;
                S_F5: begin : filt_commit
                    reg signed [31:0] y;
                    y = filt_bypass[ch] ? phys : rnd24(acc);
                    x2[ch] <= x1[ch]; x1[ch] <= phys;
                    y2[ch] <= y1[ch]; y1[ch] <= y;
                    filt[ch] <= y;
                    filt_flat[32*ch +: 32] <= y;
                    if (ch == 3'd7) begin
                        F  <= ssub(filt[0], f_offset);
                        st <= S_X0;
                    end else begin
                        ch <= ch + 3'd1;
                        st <= S_SC;
                    end
                end
                // 3-4  kinematics (F was registered above; filt[0] was final before ch 7)
                S_X0:  st <= S_X1;
                S_X1: begin : kin
                    reg signed [31:0] xn;
                    xn = rnd4(acc);
                    x <= xn;
                    if (!primed) begin x_prev <= xn; primed <= 1'b1; end
                    st <= S_V0;
                end
                S_V0:  st <= S_V1;
                S_V1: begin v_raw <= sat64(acc); st <= S_V2; end
                S_V2:  st <= S_V3;
                S_V3: begin v <= sadd(v, rnd24(acc)); x_prev <= x; st <= S_SAFE; end
                // 5-6  safety supervisor, then interaction control
                S_SAFE: begin
                    wdog    <= wdog_n;
                    latched <= latched_n;
                    f_pc    <= 0;
                    pc_active <= 1'b0;
                    if (!(en_bit && latched_n == 8'd0)) begin
                        e <= 0; x_cmd <= x; v_cmd <= 0; duty <= 0; armed <= 1'b0;
                        active_o <= 1'b0;
                        kk <= 2'd0;
                        st <= S_L0;
                    end else begin
                        if (!armed) begin e <= 0; x_cmd <= x; v_cmd <= 0; armed <= 1'b1; end
                        active_o <= 1'b1;
                        st <= S_P0;
                    end
                    fault_o <= (latched_n != 8'd0);
                end
                // 6a passivity observer
                S_P0:  st <= S_P1;
                S_P1: begin p <= rnd20(acc); st <= S_P2; end
                S_P2:  st <= S_P3;
                S_P3: begin
                    e  <= sadd(e, rnd24(acc));
                    st <= S_PC0;
                end
                // 6a passivity controller
                S_PC0: begin
                    if (pc_bit && e < 0 && sabs(v) > pc_v_eps) st <= S_PC1;
                    else                                      st <= S_A0;
                end
                S_PC1: begin v2  <= rnd20(acc); st <= S_PC2; end
                S_PC2: st <= S_PC3;
                S_PC3: begin den <= rnd24(acc); st <= S_PC4; end
                S_PC4: begin : pc_div
                    reg signed [63:0] num;
                    if (den > 0) begin
                        num = e;
                        num = -num;
                        div_n     <= num << 20;
                        div_start <= 1'b1;
                        st <= S_PC5;
                    end else begin
                        st <= S_A0;
                    end
                end
                S_PC5: if (div_done) begin
                    alpha <= (div_q > {32'd0, pc_alpha_max}) ? pc_alpha_max : div_q[31:0];
                    st <= S_PC6;
                end
                S_PC6:  st <= S_PC7;
                S_PC7:  begin f_pc <= sneg(rnd20(acc)); st <= S_PC8; end
                S_PC8:  st <= S_PC9;
                S_PC9:  begin tmp <= rnd20(acc); st <= S_PC10; end
                S_PC10: st <= S_PC11;
                S_PC11: begin e <= sadd(e, rnd24(acc)); pc_active <= 1'b1; st <= S_A0; end
                // 6b admittance
                S_A0: begin
                    err <= ssub(x_cmd, x_eq);
                    fin <= sadd(F, f_pc);
                    st  <= S_A1;
                end
                S_A1:  st <= S_A2;
                S_A2:  st <= S_A3;
                S_A3: begin v_cmd <= rnd24(acc); st <= S_A4; end
                S_A4:  st <= S_A5;
                S_A5: begin x_cmd <= sadd(x_cmd, rnd24(acc)); st <= S_I0; end
                // 6c inner loop, clamp, slew
                S_I0:  st <= S_I1;
                S_I1:  st <= S_I2;
                S_I2: begin u <= rnd24(acc); st <= S_I3; end
                S_I3:  st <= S_I4;
                S_I4: begin
                    duty <= sadd(duty, d_delta);
                    kk   <= 2'd0;
                    st   <= S_L0;
                end
                // 7  afferent relay: LIF + paired-pulse gate, four channels
                S_L0:  st <= S_L1;
                S_L1: begin lt <= ssub(lif_v[kk], rnd24(acc)); st <= S_L2; end
                S_L2:  st <= S_L3;
                S_L3: begin
                    if (lif_spike) begin
                        lif_v[kk]   <= ssub(lif_vn, lif_theta);
                        since[kk]   <= 16'd0;
                        spikes[kk]  <= 1'b1;
                        gated[kk]   <= in_gate;
                        stim_amp_out[8*kk +: 8] <= in_gate ? amp_prod[15:8] : amp_base;
                        spike_count <= spike_count + 32'd1;
                    end else begin
                        lif_v[kk] <= lif_vn;
                        if (since[kk] != 16'hFFFF) since[kk] <= since[kk] + 16'd1;
                    end
                    if (kk == 2'd3) st <= S_REP;
                    else begin kk <= kk + 2'd1; st <= S_L0; end
                end
                // 8  repetition detector
                S_REP: begin
                    if (!rep_state && x > rep_hi) rep_state <= 1'b1;
                    else if (rep_state && x < rep_lo) begin
                        rep_state <= 1'b0;
                        rep_count <= rep_count + 16'd1;
                    end
                    st <= S_D0;
                end
                // 9  outputs
                S_D0: st <= S_D1;
                S_D1: begin
                    pwm_cmp  <= acc[51:20];
                    pwm_rev  <= duty[31];
                    status   <= {latched, 4'd0, stim_bit, pc_active, (latched != 8'd0), en_bit};
                    tick_tlm <= tick_cnt;
                    tick_cnt <= tick_cnt + 32'd1;
                    done     <= 1'b1;
                    st       <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
