// apt_uart_tx.v - 8N1 UART transmitter, DIV clocks per bit.
`default_nettype none

module apt_uart_tx #(
    parameter integer DIV = 20
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       start,
    input  wire [7:0] data,
    output reg        tx,
    output reg        busy
);
    reg [9:0]  sh;
    reg [3:0]  nb;
    reg [15:0] cnt;

    always @(posedge clk) begin
        if (rst) begin
            tx <= 1'b1; busy <= 1'b0; nb <= 0; cnt <= 0;
        end else if (!busy) begin
            if (start) begin
                sh <= {1'b1, data, 1'b0};  // stop, data LSB first, start
                nb <= 4'd10; cnt <= 16'd0; busy <= 1'b1;
            end
        end else if (cnt == 16'd0) begin
            tx <= sh[0];
            sh <= {1'b1, sh[9:1]};
            cnt <= DIV[15:0] - 16'd1;
            if (nb == 4'd0) begin
                busy <= 1'b0;
                tx   <= 1'b1;
            end else nb <= nb - 4'd1;
        end else cnt <= cnt - 16'd1;
    end
endmodule

`default_nettype wire
