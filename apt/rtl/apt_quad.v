// apt_quad.v - x4 quadrature decoder with input synchronisers and an illegal-transition counter.
`default_nettype none

module apt_quad (
    input  wire               clk,
    input  wire               rst,
    input  wire               a_in,
    input  wire               b_in,
    output reg  signed [31:0] count,
    output reg  [31:0]        errors
);
    reg [2:0] a_s, b_s;  // two synchroniser flops + one history flop
    always @(posedge clk) begin
        a_s <= {a_s[1:0], a_in};
        b_s <= {b_s[1:0], b_in};
    end

    wire [1:0] prev = {a_s[2], b_s[2]};
    wire [1:0] cur  = {a_s[1], b_s[1]};

    always @(posedge clk) begin
        if (rst) begin
            count  <= 32'sd0;
            errors <= 32'd0;
        end else if (prev != cur) begin
            // Gray sequence 00 -> 01 -> 11 -> 10 -> 00 is forward (A leads B is reverse)
            case ({prev, cur})
                4'b00_01, 4'b01_11, 4'b11_10, 4'b10_00: count <= count + 32'sd1;
                4'b00_10, 4'b10_11, 4'b11_01, 4'b01_00: count <= count - 32'sd1;
                default: errors <= errors + 32'd1;  // both lines changed: a missed edge
            endcase
        end
    end
endmodule

`default_nettype wire
