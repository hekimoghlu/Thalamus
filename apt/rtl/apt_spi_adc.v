// apt_spi_adc.v - sequencer for an 8-channel simultaneous-sampling SPI ADC (AD7606-class,
// serial readout on one DOUT line).
//
//   start -> CONVST pulse -> wait CONV cycles -> wait BUSY low (timeout => fault)
//         -> CS low, clock NCH x 16 bits MSB first (sample DOUT on SCLK rising) -> done
//
// The sampling instant is the CONVST edge; the latency monitor starts its clock there.
`default_nettype none

module apt_spi_adc #(
    parameter integer NCH       = 8,
    parameter integer SCLK_HALF = 2,     // SCLK half-period in clk cycles
    parameter integer CONV      = 200,   // minimum conversion time
    parameter integer TIMEOUT   = 2000   // BUSY timeout after CONV
) (
    input  wire                clk,
    input  wire                rst,
    input  wire                start,
    output reg                 convst,
    input  wire                busy_in,
    output reg                 cs_n,
    output reg                 sclk,
    input  wire                dout,
    output reg                 done,
    output reg                 fault,
    output reg  [16*NCH-1:0]   codes     // ch(NCH-1) .. ch0
);
    localparam [2:0] IDLE = 3'd0, CONVST = 3'd1, WAIT = 3'd2, BUSYW = 3'd3, LOW = 3'd4, HIGH = 3'd5;
    localparam integer NBITS = 16 * NCH;

    reg [2:0]        st;
    reg [15:0]       cnt;
    reg [15:0]       bits;
    reg [NBITS-1:0]  sh;
    reg [1:0]        busy_s;  // synchroniser for the asynchronous BUSY pin

    always @(posedge clk) busy_s <= {busy_s[0], busy_in};

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            st <= IDLE; convst <= 1'b0; cs_n <= 1'b1; sclk <= 1'b0; fault <= 1'b0;
            codes <= {NBITS{1'b0}};
        end else begin
            case (st)
                IDLE: if (start) begin
                    convst <= 1'b1; cnt <= 16'd3; st <= CONVST;
                end
                CONVST: if (cnt == 0) begin
                    convst <= 1'b0; cnt <= CONV[15:0]; st <= WAIT;
                end else cnt <= cnt - 16'd1;
                WAIT: if (cnt == 0) begin
                    cnt <= TIMEOUT[15:0]; st <= BUSYW;
                end else cnt <= cnt - 16'd1;
                BUSYW: if (!busy_s[1]) begin
                    fault <= 1'b0;
                    cs_n <= 1'b0; bits <= NBITS[15:0]; cnt <= SCLK_HALF[15:0] - 16'd1; st <= LOW;
                end else if (cnt == 0) begin
                    fault <= 1'b1; done <= 1'b1; st <= IDLE;   // keep the previous codes
                end else cnt <= cnt - 16'd1;
                LOW: if (cnt == 0) begin
                    sclk <= 1'b1; sh <= {sh[NBITS-2:0], dout};
                    cnt <= SCLK_HALF[15:0] - 16'd1; st <= HIGH;
                end else cnt <= cnt - 16'd1;
                HIGH: if (cnt == 0) begin
                    sclk <= 1'b0;
                    if (bits == 16'd1) begin
                        cs_n <= 1'b1;
                        // the first word clocked out is ch0
                        begin : unpack
                            integer c;
                            for (c = 0; c < NCH; c = c + 1)
                                codes[16*c +: 16] <= sh[NBITS-1-16*c -: 16];
                        end
                        done <= 1'b1; st <= IDLE;
                    end else begin
                        bits <= bits - 16'd1; cnt <= SCLK_HALF[15:0] - 16'd1; st <= LOW;
                    end
                end else cnt <= cnt - 16'd1;
                default: st <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
