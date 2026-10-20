// apt_udiv.v - 64/32 unsigned restoring divider, one quotient bit per cycle (64 cycles).
// Used by the passivity controller: alpha = (-E << 20) / (v^2 dt). Truncating division, the
// same as C++ unsigned integer division in the golden model.
`default_nettype none

module apt_udiv (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [63:0] dividend,
    input  wire [31:0] divisor,
    output reg         busy,
    output reg         done,
    output reg  [63:0] quotient
);
    reg [63:0] n;
    reg [32:0] r;
    reg [31:0] d;
    reg [6:0]  cnt;
    reg [32:0] rs;

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            busy <= 1'b0;
            cnt  <= 7'd0;
        end else if (start) begin
            n        <= dividend;
            d        <= divisor;
            r        <= 33'd0;
            quotient <= 64'd0;
            cnt      <= 7'd64;
            busy     <= 1'b1;
        end else if (busy) begin
            rs = {r[31:0], n[63]};
            if (rs >= {1'b0, d}) begin
                r        <= rs - {1'b0, d};
                quotient <= {quotient[62:0], 1'b1};
            end else begin
                r        <= rs;
                quotient <= {quotient[62:0], 1'b0};
            end
            n   <= {n[62:0], 1'b0};
            cnt <= cnt - 7'd1;
            if (cnt == 7'd1) begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
