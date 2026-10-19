// apt_telemetry.v - per-tick telemetry framer: 42-byte payload, CRC-16/CCITT-FALSE (little-
// endian), SLIP framing (RFC 1055), 8N1 UART. Byte layout matches link::Telemetry::pack().
//
// The payload is snapshotted on `send`; a `send` that arrives while a frame is still on the wire
// is dropped and counted rather than corrupting the frame in flight.
`default_nettype none

module apt_telemetry #(
    parameter integer UART_DIV = 20
) (
    input  wire         clk,
    input  wire         rst,
    input  wire         send,
    input  wire [335:0] payload,   // byte i at [8i +: 8]
    output wire         tx,
    output reg  [31:0]  dropped,
    output wire         busy
);
    localparam integer NPAY = 42, NTOT = 44;  // payload + CRC
    localparam [7:0] END_ = 8'hC0, ESC = 8'hDB, ESC_END = 8'hDC, ESC_ESC = 8'hDD;

    function [15:0] crc_byte;  // CRC-16/CCITT-FALSE, poly 0x1021, MSB first
        input [15:0] c;
        input [7:0]  b;
        integer k;
        reg [15:0] r;
        begin
            r = c ^ {b, 8'h00};
            for (k = 0; k < 8; k = k + 1) r = r[15] ? ((r << 1) ^ 16'h1021) : (r << 1);
            crc_byte = r;
        end
    endfunction

    // UART transmitter
    reg        u_start;
    reg  [7:0] u_data;
    wire       u_busy;
    apt_uart_tx #(.DIV(UART_DIV)) u_tx (
        .clk(clk), .rst(rst), .start(u_start), .data(u_data), .tx(tx), .busy(u_busy)
    );

    localparam [2:0] IDLE = 3'd0, SOF = 3'd1, BYTE = 3'd2, ESC2 = 3'd3, EOF_ = 3'd4, WAITU = 3'd5;
    reg [2:0]   st, ret;
    reg [335:0] pl;
    reg [5:0]   idx;
    reg [15:0]  crc;
    reg [7:0]   esc_byte;

    assign busy = (st != IDLE);

    wire [7:0] cur = (idx < NPAY) ? pl[8*idx +: 8] : (idx == NPAY ? crc[7:0] : crc[15:8]);

    always @(posedge clk) begin
        u_start <= 1'b0;
        if (rst) begin
            st <= IDLE; dropped <= 0; idx <= 0; crc <= 16'hFFFF;
        end else begin
            if (send && st != IDLE) dropped <= dropped + 32'd1;
            case (st)
                IDLE: if (send) begin
                    pl <= payload; idx <= 0; crc <= 16'hFFFF; st <= SOF;
                end
                SOF: if (!u_busy && !u_start) begin
                    u_data <= END_; u_start <= 1'b1; ret <= BYTE; st <= WAITU;
                end
                BYTE: if (!u_busy && !u_start) begin
                    if (idx == NTOT[5:0]) begin
                        st <= EOF_;
                    end else begin
                        if (idx < NPAY) crc <= crc_byte(crc, cur);
                        if (cur == END_ || cur == ESC) begin
                            u_data <= ESC; esc_byte <= (cur == END_) ? ESC_END : ESC_ESC; ret <= ESC2;
                        end else begin
                            u_data <= cur; ret <= BYTE;
                        end
                        u_start <= 1'b1;
                        idx <= idx + 6'd1;
                        st <= WAITU;
                    end
                end
                ESC2: if (!u_busy && !u_start) begin
                    u_data <= esc_byte; u_start <= 1'b1; ret <= BYTE; st <= WAITU;
                end
                EOF_: if (!u_busy && !u_start) begin
                    u_data <= END_; u_start <= 1'b1; ret <= IDLE; st <= WAITU;
                end
                WAITU: if (u_busy) st <= ret;  // transmitter accepted the byte
                default: st <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
