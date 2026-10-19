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

    always @(posedge clk) begin
        onset   <= 1'b0;
        overrun <= 1'b0;
        if (rst) begin
            st <= IDLE; cath <= 1'b0; anod <= 1'b0; dac <= 8'd0; cnt <= 0;
        end else begin
            if (trigger && enable && st != IDLE) overrun <= 1'b1;
            case (st)
                IDLE: if (trigger && enable && amp != 8'd0) begin
                    dac <= amp; cath <= 1'b1; onset <= 1'b1;
                    cnt <= (width == 0) ? 32'd0 : width - 32'd1; st <= PH1;
                end
                PH1: if (cnt == 0) begin
                    cath <= 1'b0; cnt <= gap; st <= GAPS;
                end else cnt <= cnt - 32'd1;
                GAPS: if (cnt == 0) begin
                    anod <= 1'b1; cnt <= (width == 0) ? 32'd0 : width - 32'd1; st <= PH2;
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
endmodule

`default_nettype wire
