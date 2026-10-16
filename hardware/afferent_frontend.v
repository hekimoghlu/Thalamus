// Copyright (c) 2026 Omnira CJSC. All rights reserved.
// Author: Tunjay Akbarli
// Date: August 27, 2026

// Causal event detector and latency gate for a closed-loop afferent interface.

module afferent_frontend #(
    parameter DATA_W    = 12,                 // sensor ADC width
    parameter SHIFT     = 2,                  // IIR smoothing, alpha = 2^-SHIFT
    parameter CLK_HZ    = 50_000_000,
    parameter BUDGET_NS = 10_000_000          // 10 ms loop budget
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              sample_valid,    // one strobe per sensor sample
    input  wire [DATA_W-1:0] sample,          // e.g. grip force or joint angle
    input  wire [DATA_W-1:0] threshold,       // event threshold
    input  wire              cmd_ack,         // actuator accepted the command
    output reg               cmd_valid,       // one-cycle command strobe on event
    output reg  [31:0]       latency_cycles,  // event-to-ack latency of last command
    output reg               budget_violation // latency exceeded the budget
);
    localparam W = DATA_W + SHIFT;
    localparam integer BUDGET_CYCLES = (BUDGET_NS / 1000) * (CLK_HZ / 1000) / 1000;

    reg [W-1:0]  y;          // filter state, scaled by 2^SHIFT
    reg          above;      // filtered signal was above threshold last sample
    reg          pending;    // command issued, waiting for ack
    reg [31:0]   lat_cnt;

    // y[n] = y[n-1] + alpha * (x[n] - y[n-1]), unsigned-safe form
    wire [W-1:0] xs     = {sample, {SHIFT{1'b0}}};
    wire [W-1:0] y_next = (xs >= y) ? (y + ((xs - y) >> SHIFT))
                                    : (y - ((y - xs) >> SHIFT));
    wire [DATA_W-1:0] yf = y[W-1:SHIFT];

    wire event_pulse = sample_valid && (yf >= threshold) && !above;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y <= 0; above <= 0; pending <= 0; lat_cnt <= 0;
            cmd_valid <= 0; latency_cycles <= 0; budget_violation <= 0;
        end else begin
            cmd_valid <= 1'b0;
            if (sample_valid) begin
                y     <= y_next;
                above <= (yf >= threshold);
            end
            if (event_pulse) begin
                cmd_valid <= 1'b1;
                pending   <= 1'b1;
                lat_cnt   <= 0;
            end else if (pending) begin
                lat_cnt <= lat_cnt + 1'b1;
            end
            if (pending && cmd_ack) begin
                pending          <= 1'b0;
                latency_cycles   <= lat_cnt + 1'b1;
                budget_violation <= (lat_cnt + 1'b1 > BUDGET_CYCLES);
            end
        end
    end
endmodule
