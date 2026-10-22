// apt_interlock.v - the combinational safety interlock between the datapath and the drivers.
//
// Bridge drive needs an active session, no latched fault and no e-stop; stimulation needs its
// CTRL bit, no latched fault and no e-stop. Both are pure functions of their inputs, so the
// e-stop pin (after the 2-flop synchroniser) reaches the gate pins in exactly one more clock.
// Kept as its own module so formal/safety_chain.sby can prove the whole chain
// interlock -> PWM / stimulator without elaborating the datapath.
`default_nettype none
module apt_interlock (
    input  wire core_active,
    input  wire core_fault,
    input  wire estop,
    input  wire stim_ctrl,
    output wire drive_en,
    output wire stim_en
);
    assign drive_en = core_active && !core_fault && !estop;
    assign stim_en  = stim_ctrl   && !core_fault && !estop;
endmodule
`default_nettype wire
