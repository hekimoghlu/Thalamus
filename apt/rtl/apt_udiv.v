// apt_udiv.v - NxM unsigned restoring divider (default 64/32), one quotient bit per cycle.
// Used by the passivity controller: alpha = (-E << 20) / (v^2 dt). Truncating division, the
// same as C++ unsigned integer division in the golden model.
`default_nettype none

module apt_udiv #(
    parameter integer N = 64,   // dividend / quotient width
    parameter integer M = 32    // divisor width
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [N-1:0] dividend,
    input  wire [M-1:0] divisor,
    output reg         busy,
    output reg         done,
    output reg  [N-1:0] quotient
);
    reg [N-1:0] n;
    reg [M:0]   r;
    reg [M-1:0] d;
    reg [6:0]  cnt;
    reg [M:0]   rs;

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            busy <= 1'b0;
            cnt  <= 7'd0;
        end else if (start) begin
            n        <= dividend;
            d        <= divisor;
            r        <= 0;
            quotient <= 0;
            cnt      <= N[6:0];
            busy     <= 1'b1;
        end else if (busy) begin
            rs = {r[M-1:0], n[N-1]};
            if (rs >= {1'b0, d}) begin
                r        <= rs - {1'b0, d};
                quotient <= {quotient[N-2:0], 1'b1};
            end else begin
                r        <= rs;
                quotient <= {quotient[N-2:0], 1'b0};
            end
            n   <= {n[N-2:0], 1'b0};
            cnt <= cnt - 7'd1;
            if (cnt == 7'd1) begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        end
    end

`ifdef FORMAL
    // Restoring-division invariant: after k steps, with N0 the dividend and D the divisor,
    //   N0 = (quotient * D + r) * 2^(64-k) + (remaining dividend bits)
    // so at completion quotient * D + r = N0 and r < D (for D != 0).  Checked with a ghost copy.
    localparam integer W = N + M + 2;
    reg [N-1:0] g_n0;
    reg [M-1:0] g_d0;
    reg        f_init = 1'b0, g_active = 1'b0;
    always @(posedge clk) begin
        f_init <= 1'b1;
        if (!f_init) assume(rst);
        if (!rst && start && !busy) begin g_n0 <= dividend; g_d0 <= divisor; end
        if (rst) g_active <= 1'b0; else if (start && !busy) g_active <= 1'b1;
        // a start while busy would be a protocol violation by the caller
        assume(!(start && busy));
        if (f_init && !rst) begin assert(!busy || g_active); assert(!done || (g_active && !busy && cnt == 7'd0)); end
        if (f_init && !rst && g_active) begin
            // d is stable and never changes during a division
            assert(d == g_d0);
            if (busy) begin
                assert(cnt >= 7'd1 && cnt <= N[6:0]);
                assert(n == (g_n0 << (N[6:0] - cnt)));
                // quotient*D + r  ==  top (64-cnt) bits of the dividend
                if (g_d0 != 0) assert(({{(W-N){1'b0}}, quotient} * {{(W-M){1'b0}}, d} + {{(W-M-1){1'b0}}, r}) == ({{(W-N){1'b0}}, g_n0} >> cnt));
                if (g_d0 != 0) assert(r < {1'b0, d});
            end
        end
        if (f_init && !rst && done && g_d0 != 0) begin
            assert({{(W-N){1'b0}}, quotient} * {{(W-M){1'b0}}, d} + {{(W-M-1){1'b0}}, r} == {{(W-N){1'b0}}, g_n0});
            assert(r < {1'b0, d});
        end
    end
`endif
endmodule

`default_nettype wire
