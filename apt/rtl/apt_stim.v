// apt_stim.v - charge-balanced biphasic pulse generator for one afferent (electrotactile) channel.
//
//   trigger -> cathodic phase (WIDTH) -> interphase gap (GAP) -> anodic phase (WIDTH) -> refractory
//
// Both phases carry the same amplitude code for the same number of cycles, so every pulse that
// starts is charge-balanced by construction - including when `enable` drops mid-pulse: the pulse
// in flight completes, and only new triggers are refused. A trigger that arrives while a pulse or
// its refractory period is in progress is dropped and counted.
`default_nettype none

module apt_stim (
    input  wire        clk,
    input  wire        rst,
    input  wire        enable,
    input  wire        trigger,
    input  wire [7:0]  amp,
    input  wire [31:0] width,
    input  wire [31:0] gap,
    input  wire [31:0] refract,
    output reg         cath,      // cathodic phase drive
    output reg         anod,      // anodic phase drive
    output reg  [7:0]  dac,       // current amplitude code
    output reg         onset,     // pulses on the first cycle of the cathodic phase
    output reg         overrun
);
    localparam [2:0] IDLE = 3'd0, PH1 = 3'd1, GAPS = 3'd2, PH2 = 3'd3, REF = 3'd4;
    reg [2:0]  st;
    reg [31:0] cnt;
    reg [31:0] wl;            // phase width latched at trigger: both phases use the same value

    always @(posedge clk) begin
        onset   <= 1'b0;
        overrun <= 1'b0;
        if (rst) begin
            st <= IDLE; cath <= 1'b0; anod <= 1'b0; dac <= 8'd0; cnt <= 0; wl <= 32'd1;
        end else begin
            if (trigger && enable && st != IDLE) overrun <= 1'b1;
            case (st)
                IDLE: if (trigger && enable && amp != 8'd0) begin
                    dac <= amp; cath <= 1'b1; onset <= 1'b1; wl <= (width == 0) ? 32'd1 : width;
                    cnt <= (width == 0) ? 32'd0 : width - 32'd1; st <= PH1;
                end
                PH1: if (cnt == 0) begin
                    cath <= 1'b0; cnt <= gap; st <= GAPS;
                end else cnt <= cnt - 32'd1;
                GAPS: if (cnt == 0) begin
                    anod <= 1'b1; cnt <= wl - 32'd1; st <= PH2;
                end else cnt <= cnt - 32'd1;
                PH2: if (cnt == 0) begin
                    anod <= 1'b0; dac <= 8'd0; cnt <= refract; st <= REF;
                end else cnt <= cnt - 32'd1;
                REF: if (cnt == 0) st <= IDLE;
                     else cnt <= cnt - 32'd1;
                default: st <= IDLE;
            endcase
        end
    end

`ifdef FORMAL
    // Ghost net charge (cathodic cycles minus anodic cycles). Together with the phase invariants
    // below, induction proves it is zero whenever no pulse is in flight - for every width/gap/
    // refractory value the CSR could ever hold, including values rewritten mid-pulse.
    reg signed [34:0] net;
    reg               f_init = 1'b0;
    wire [34:0] wl35 = {3'd0, wl}, cnt35 = {3'd0, cnt};
    always @(posedge clk) begin
        f_init <= 1'b1;
        if (!f_init) assume(rst);
        if (rst) net <= 0; else net <= net + $signed({34'd0, cath}) - $signed({34'd0, anod});
        if (f_init) begin
            assert(wl != 0);
            assert(st <= REF);
            assert(cath == (st == PH1));
            assert(anod == (st == PH2));
        end
    end
    // properties that hold from the first post-reset state (guarded by past-valid)
    reg f_past = 1'b0;
    always @(posedge clk) begin
        f_past <= f_init && !rst;
        if (f_past) begin
            assert(st != PH1 || (cnt < wl && net == (wl35 - 1 - cnt35)));
            assert(st != GAPS || net == wl35);
            assert(st != PH2 || (cnt < wl && net == cnt35 + 1));
            assert(st != IDLE || net == 0);
            assert(st != REF  || net == 0);
        end
    end
`endif
endmodule

`default_nettype wire
