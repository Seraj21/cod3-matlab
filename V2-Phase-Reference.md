# Effective-channel phase-reference diagnostic

This is a controlled diagnostic, not a replacement of the frozen V2 comparison.

The user-supplied *Practical Hybrid Beamforming Schemes in Massive MIMO 5G NR Systems*, PDF page 3, equations (1)-(2), writes the physical model as `y[k] = H[k] VRF Vdig[k] s[k] + z[k]` and the channel estimated by the UE as `Heff[k] = H[k] VRF Vdig[k]`. *Massive MIMO Networks*, section 4.3.3, similarly distinguishes the precoded scalar channel that the UE must learn from the propagation channel used to design the precoder. Neither source is claimed to propose the following diagnostic.

For a unit-modulus diagonal matrix D of stream phases, replace B by B D. Every quantity `abs(H(i,:)*B(:,u))^2` and each column norm is unchanged. This holds for an arbitrary actual channel, not only the estimate used to select D. It is therefore possible to change the phase reference without increasing the nominal desired or interfering powers. A frequency-selective, time-varying waveform can still exhibit different ICI/ISI or nonlinear RF distortion; actual waveform payload tests remain necessary.

`v2PhaseGauge` derives D from the estimated desired response, never from the true channel in an operational method. Two cases are recorded separately:

- **Quantized anchor:** the common rotation is restricted to the same 16-point phase lattice. Both absolute and relative 4-bit phase constraints are preserved.
- **Digital phase anchor:** one arbitrary common digital phase is applied per stream/resource group after the 4-bit relative phases. Relative phase differences remain on the 16-point lattice, but the absolute baseband coefficient phases are no longer restricted to that lattice. This additional digital freedom is explicit and is not presented as the exact same absolute constraint.

Both cases preserve constant coefficient modulus, equal user power, four users, 512-QAM, LDPC, TDL-C, radio impairments, and existing reference resources. Beam widths 1 and 4 are identified. Practical methods receive the same channel and thermal-noise realization. The perfect receiver is an explicitly labeled diagnostic oracle.

Regression tests verify beam-response powers, total power, the phase lattice, and the resulting predicted desired-channel phase, including single-precision stored input weights. The two-frame waveform control uses new independent seeds and the old frozen receiver models. Its purpose is to diagnose a phase/pilot representation problem; results are not a final retrained-CNN performance claim.
