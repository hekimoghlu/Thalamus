// Formal wrapper: interlock -> H-bridge and stimulator, all datapath inputs free.
module safety_chain (
    input clk, input rst,
    input core_active, input core_fault, input estop, input stim_ctrl,
    input update, input [31:0] cmp_in, input rev_in,
    input trigger, input [7:0] amp, input [31:0] width, input [31:0] gap, input [31:0] refract,
    output ha, la, hb, lb, onset, cath, anod
);
    wire drive_en, stim_en, loaded, overrun;
    wire [7:0] dac;
    apt_interlock ilk (.core_active(core_active), .core_fault(core_fault), .estop(estop),
                       .stim_ctrl(stim_ctrl), .drive_en(drive_en), .stim_en(stim_en));
    apt_pwm #(.PERIOD(1250), .DEADTIME(25)) pwm (.clk(clk), .rst(rst), .enable(drive_en), .update(update),
        .cmp_in(cmp_in), .rev_in(rev_in), .loaded(loaded), .ha(ha), .la(la), .hb(hb), .lb(lb));
    apt_stim stim (.clk(clk), .rst(rst), .enable(stim_en), .trigger(trigger), .amp(amp), .width(width),
        .gap(gap), .refract(refract), .cath(cath), .anod(anod), .dac(dac), .onset(onset), .overrun(overrun));

    reg f_init = 1'b0, f_past = 1'b0, p_estop = 1'b0, p_fault = 1'b0;
    always @(posedge clk) begin
        f_init <= 1'b1; f_past <= f_init && !rst; p_estop <= estop; p_fault <= core_fault;
        if (!f_init) assume(rst);
        if (f_past) begin
            // e-stop or a latched fault: every gate off and no new pulse on the next clock
            if (p_estop || p_fault) begin
                assert(!{ha, la, hb, lb});
                assert(!onset);
            end
        end
    end
endmodule
