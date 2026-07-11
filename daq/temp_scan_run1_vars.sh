imgFOLDER=/data/images/2025-01-10_check
runname=minos_2025-01-10_135K_check
#runname=moduleC40_41-ssc16_17-lta20_60_TEMP135K-run2

#daemonport variable is used by setup_lta.sh
daemonport=8888

#the loop script creates the lockfile; delete it to stop the loop (after the current image is done)
lockfilename=lockfile

#e-mail for alert messages
email=suemura@fnal.gov

clearseq=sequencers/C/sequencer_clear_C.xml
imgseq=sequencers/C/sequencer_C_skip_noreset.xml
#imgseq=sequencers/C/sequencer_C_skip_noreset_sideU.xml
#imgseq=sequencers/C/sequencer_C.xml
#exposeseq=sequencers/C/sequencer_C_expose_binned_noreset.xml
exposeseq=sequencers/C/sequencer_C_expose_binned_noreset_clockdroop.xml

#initscript=init/init_skp_lta_v2_smart.sh
initscript=init/init_skp_lta_v2_smart_multi.sh
voltagescript=voltage_skp_lta_v2_C_minos.sh
#voltagescript=voltage_skp_lta_v2_zero.sh
# for astro test
#voltagescript=voltage_skp_lta_v2_C_safe.sh
#voltagescript=voltage_skp_lta_v3_AstroSkipper.sh
# for replicating "run34" spurious charge
#voltagescript=minos_2022-11-16_135K_run34_voltage.sh

cdsout=3 #ped-sig: for SSC
#cdsout=2 #sig-ped: for mistica

VSUB=70
#VSUB=20

cdsSAMP=250
cdsSINIT=40
seqPEDEXTRA=70
seqSIGEXTRA=5

#image dimensions before binning
totNCOL=3200
#totNCOL=6400
#totNCOL=4000
#totNROW=520
totNROW=640
#totNROW=200
#totNROW=20
#totNROW=50

#NSAMP is overridden by some scripts (sho_nsamp1.sh, sho_noise.sh)
skpNSAMP=400
#skpNSAMP=800
