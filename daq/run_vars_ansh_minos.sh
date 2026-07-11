imgFOLDER=/data/images/2025-03-18/
#runname=readout_optimize_run5
#runname=readout_baseline_run3
runname=temp_scan_run1
#daemonport variable is used by setup_lta.sh
daemonport=8888

#the loop script creates the lockfile; delete it to stop the loop (after the current image is done)
lockfilename=lockfile_ansh
email=adesai@uoregon.edu
clearseq=sequencers/C/sequencer_clear_C.xml
imgseq=sequencers/C/sequencer_C_skip_noreset_sr.xml
exposeseq=sequencers/C/sequencer_C_expose_binned_noreset.xml 
skipseq=sequencers/C/sequencer_C_expose_binned_noreset.xml
scseq=sequencers/C/sc_gen_sequencer.xml
# scimgseq=sequencers/C/sc_gen_skip_noreset.xml
# metascseq=sequencers/C/meta_sc_gen_sequencer.xml
# pumpseq=sequencers/sequencer_microchip_ppump_ph1and3_optimal.xml
pumpseq=sequencers/C/pocket_pumping_img.xml
pumponlyseq=sequencers/C/pocket_pumping.xml

#skipseq=sequencers/C/sequencer_C_expose_expose_and_skip_noreset.xml

initscript=init/init_skp_lta_v2_smart_multi.sh
# initscript=init/init_skp_lta_v2_smart.sh
# voltagescript=voltage_skp_lta_v2_C_minos_ansh.sh
spuriousvoltagescript=voltage_skp_lta_v2_C_minos_scgen.sh
voltagescript=voltage_skp_lta_v2_C_minos.sh
# sloshscript=voltage_skp_lta_v2_C_minos_ansh_slosh.sh
highervoltagescript=voltage_skp_lta_v2_C_minos_higherreadout.sh

cdsout=3 #ped-sig: for SSC
# cdsout=2 #sig-ped: for mistica

VSUB=70

cdsSAMP=200
#not "noreset"
# cdsSINIT=30
# seqPEDEXTRA=140
# seqSIGEXTRA=100
#for "noreset"
cdsSINIT=40
seqPEDEXTRA=70
seqSIGEXTRA=5

skpNCOL=500

#image dimensions before binning
itotNCOL=3500
totNROW=600

#NSAMP is overridden by some scripts (sho_nsamp1.sh, sho_noise.sh)
skpNSAMP=1

skpTemp=230


