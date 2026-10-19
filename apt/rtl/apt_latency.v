// apt_latency.v - measures the two loop latencies the manuscript's envelopes are drawn against,
// on the hardware itself, every tick:
//
//   motor     sampling instant (CONVST) -> PWM shadow register applied
//   afferent  sampling instant (CONVST) -> first stimulus onset caused by that tick
//
// A motor latency above BUDGET (when BUDGET != 0) raises `overrun`, which the core latches as a
// LATENCY fault on the next tick.
`default_nettype none

module apt_latency (
    input  wire        clk,
    input  wire        rst,
    input  wire        sample,        // CONVST
    input  wire        core_done,
    input  wire        any_spike,     // core produced at least one afferent event this tick
    input  wire        pwm_loaded,
    input  wire        stim_onset,
    input  wire [31:0] budget,
    output reg  [31:0] motor_last,
    output reg  [31:0] motor_max,
    output reg  [31:0] aff_last,
    output reg  [31:0] aff_max,
    output reg         overrun
);
    reg [31:0] t;
    reg        motor_open, aff_open, aff_armed, awaiting_load;

    always @(posedge clk) begin
        overrun <= 1'b0;
        if (rst) begin
            t <= 0; motor_open <= 1'b0; aff_open <= 1'b0; aff_armed <= 1'b0; awaiting_load <= 1'b0;
            motor_last <= 0; motor_max <= 0; aff_last <= 0; aff_max <= 0;
        end else begin
            if (t != 32'hFFFFFFFF) t <= t + 32'd1;
            if (sample) begin
                t <= 32'd0; motor_open <= 1'b1; aff_open <= 1'b1; aff_armed <= 1'b0;
                awaiting_load <= 1'b0;
            end
            if (core_done && motor_open) begin
                awaiting_load <= 1'b1;
                aff_armed     <= any_spike;
                if (!any_spike) aff_open <= 1'b0;
            end
            // the PWM applies the new compare at the first counter bottom after `done`
            if (pwm_loaded && awaiting_load && motor_open) begin
                motor_last <= t;
                if (t > motor_max) motor_max <= t;
                if (budget != 32'd0 && t > budget) overrun <= 1'b1;
                motor_open <= 1'b0;
                awaiting_load <= 1'b0;
            end
            if (stim_onset && aff_armed && aff_open) begin
                aff_last <= t;
                if (t > aff_max) aff_max <= t;
                aff_open <= 1'b0;
                aff_armed <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
