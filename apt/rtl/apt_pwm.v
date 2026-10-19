// apt_pwm.v - centre-aligned PWM for a full H-bridge, sign-magnitude, with dead-time insertion.
//
// The compare value and direction are taken from shadow registers at the counter's bottom, so a
// duty update never produces a runt pulse. `loaded` pulses on the cycle the shadow is applied:
// that is the actuation instant the latency monitor stops its clock on. With `enable` low all
// four gates are held off (coast) from the next cycle, without waiting for the period boundary.
`default_nettype none

module apt_pwm #(
    parameter integer PERIOD   = 1250,   // counter peak; PWM frequency = clk / (2 PERIOD)
    parameter integer DEADTIME = 25
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        enable,
    input  wire        update,          // new shadow values below
    input  wire [31:0] cmp_in,
    input  wire        rev_in,
    output reg         loaded,
    output reg         ha, la, hb, lb   // high/low gate drives, legs A and B
);
    reg [15:0] cnt;
    reg        dir_up;
    reg [31:0] cmp_sh, cmp;
    reg        rev_sh, rev, pending;

    always @(posedge clk) begin
        loaded <= 1'b0;
        if (rst) begin
            cnt <= 16'd0; dir_up <= 1'b1; cmp <= 0; cmp_sh <= 0; rev <= 1'b0; rev_sh <= 1'b0;
            pending <= 1'b0;
        end else begin
            if (update) begin cmp_sh <= cmp_in; rev_sh <= rev_in; pending <= 1'b1; end
            if (dir_up) begin
                if (cnt == PERIOD[15:0] - 16'd1) dir_up <= 1'b0;
                cnt <= cnt + 16'd1;
            end else begin
                if (cnt == 16'd1) dir_up <= 1'b1;
                cnt <= cnt - 16'd1;
            end
            if (cnt == 16'd0 && pending && !update) begin
                cmp <= cmp_sh; rev <= rev_sh; pending <= 1'b0; loaded <= 1'b1;
            end
        end
    end

    // active leg: switches; the other leg: low side held on (sign-magnitude, slow decay)
    wire pwm_on = ({16'd0, cnt} < cmp);
    wire want_ha = enable && !rev && pwm_on;
    wire want_la = enable && (rev || !pwm_on);
    wire want_hb = enable && rev && pwm_on;
    wire want_lb = enable && (!rev || !pwm_on);

    // dead-time: a gate may turn on only after its complement has been off for DEADTIME cycles
    reg [15:0] off_ha, off_la, off_hb, off_lb;
    always @(posedge clk) begin
        if (rst) begin
            {ha, la, hb, lb} <= 4'b0000;
            off_ha <= 0; off_la <= 0; off_hb <= 0; off_lb <= 0;
        end else begin
            off_ha <= ha ? 16'd0 : (off_ha == 16'hFFFF ? off_ha : off_ha + 16'd1);
            off_la <= la ? 16'd0 : (off_la == 16'hFFFF ? off_la : off_la + 16'd1);
            off_hb <= hb ? 16'd0 : (off_hb == 16'hFFFF ? off_hb : off_hb + 16'd1);
            off_lb <= lb ? 16'd0 : (off_lb == 16'hFFFF ? off_lb : off_lb + 16'd1);
            ha <= want_ha && !la && off_la >= DEADTIME[15:0];
            la <= want_la && !ha && off_ha >= DEADTIME[15:0];
            hb <= want_hb && !lb && off_lb >= DEADTIME[15:0];
            lb <= want_lb && !hb && off_hb >= DEADTIME[15:0];
        end
    end
endmodule

`default_nettype wire
