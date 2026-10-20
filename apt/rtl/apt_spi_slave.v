// apt_spi_slave.v - SPI mode-0 slave for the CSR bus (supervisory MCU -> FPGA).
//
// One transaction, CS low for exactly 40 SCLK cycles, MSB first:
//   byte 0  [7] = 1 write / 0 read, [6:0] = word address
//   bytes 1-4  data (write: from MOSI; read: on MISO, starting on the falling edge after bit 8)
// SCLK, MOSI and CS are oversampled through two-flop synchronisers, so SCLK must be at most
// clk / 8. Writes commit on CS rising, only if exactly 40 bits were clocked.
`default_nettype none

module apt_spi_slave (
    input  wire        clk,
    input  wire        rst,
    input  wire        sclk,
    input  wire        cs_n,
    input  wire        mosi,
    output reg         miso,
    output reg  [6:0]  addr,
    output reg         rd,          // pulse: address valid; rdata is sampled two cycles later
    input  wire [31:0] rdata,
    output reg         wr,          // pulse: addr/wdata valid
    output reg  [31:0] wdata
);
    reg [2:0] sclk_s, cs_s;
    reg [1:0] mosi_s;
    always @(posedge clk) begin
        sclk_s <= {sclk_s[1:0], sclk};
        cs_s   <= {cs_s[1:0], cs_n};
        mosi_s <= {mosi_s[0], mosi};
    end
    wire sclk_rise = (sclk_s[2:1] == 2'b01);
    wire sclk_fall = (sclk_s[2:1] == 2'b10);
    wire cs_active = !cs_s[1];
    wire cs_rise   = (cs_s[2:1] == 2'b01);

    reg [5:0]  nbits;
    reg [39:0] shin;
    reg [31:0] shout;
    reg [1:0]  load_pending;  // the CSR read mux is registered: wait for it

    always @(posedge clk) begin
        rd <= 1'b0;
        wr <= 1'b0;
        if (rst) begin
            nbits <= 0; miso <= 1'b0; load_pending <= 2'b00; addr <= 0; wdata <= 0;
        end else begin
            load_pending <= {load_pending[0], 1'b0};
            if (load_pending[1]) shout <= rdata;
            if (!cs_active) begin
                nbits <= 0;
                if (cs_rise && nbits == 6'd40 && shin[39]) begin
                    addr <= shin[38:32]; wdata <= shin[31:0]; wr <= 1'b1;
                end
            end else begin
                if (sclk_rise) begin
                    shin  <= {shin[38:0], mosi_s[1]};
                    if (nbits != 6'd63) nbits <= nbits + 6'd1;
                    if (nbits == 6'd7) begin
                        addr <= {shin[5:0], mosi_s[1]};
                        rd   <= 1'b1;
                        load_pending <= 2'b01;
                    end
                end
                if (sclk_fall) begin
                    if (nbits >= 6'd8 && nbits < 6'd40) begin
                        miso  <= shout[31];
                        shout <= {shout[30:0], 1'b0};
                    end else begin
                        miso <= 1'b0;
                    end
                end
            end
        end
    end
endmodule

`default_nettype wire
