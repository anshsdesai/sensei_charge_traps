#!/bin/bash

source run_vars_ansh_minos.sh

mkdir -p $imgFOLDER

cp run_vars.sh ${imgFOLDER}/${runname}_vars.sh
cp $BASH_SOURCE ${imgFOLDER}/${runname}.sh
cp $exposeseq ${imgFOLDER}/${runname}_exposeseq.xml
cp $imgseq ${imgFOLDER}/${runname}_imgseq.xml
cp $clearseq ${imgFOLDER}/${runname}_clearseq.xml
cp $scseq ${imgFOLDER}/${runname}_scseq.xml
cp $pumpseq ${imgFOLDER}/${runname}_pumpseq.xml


cp skipper_all.sh ${imgFOLDER}/skipper_all.sh
cp cal_all.sh ${imgFOLDER}/cal_all.sh

chmod 777 ${imgFOLDER}/cal_all.sh
chmod 777 ${imgFOLDER}/skipper_all.sh

source $initscript

lta set cdsout $cdsout

touch $lockfilename


doSkipper() {
    lta set psamp $cdsSAMP
    lta set ssamp $cdsSAMP
    lta set pinit 0
    lta set sinit $cdsSINIT
    let "seqINTPED = cdsSAMP + seqPEDEXTRA"
    let "seqINTSIG = cdsSAMP + cdsSINIT + seqSIGEXTRA"
    lta delay_Integ_ped $seqINTPED
    lta delay_Integ_sig $seqINTSIG

    lta NCOL $skpNCOL
    lta NROW $skpNROW
    lta NSAMP $skpNSAMP
    lta NBINROW $skpNBINROW
    lta NBINCOL $skpNBINCOL
    lta SKIPROW $skpSKIPROW
    lta SKIPCOL $skpSKIPCOL

    lta read
}

doRunseq() {
    lta set psamp $cdsSAMP
    lta set ssamp $cdsSAMP
    lta set pinit 0
    lta set sinit $cdsSINIT
    let "seqINTPED = cdsSAMP + seqPEDEXTRA"
    let "seqINTSIG = cdsSAMP + cdsSINIT + seqSIGEXTRA"
    lta delay_Integ_ped $seqINTPED
    lta delay_Integ_sig $seqINTSIG

    lta NCOL $skpNCOL
    lta NROW $skpNROW
    lta NSAMP $skpNSAMP
    lta NBINROW $skpNBINROW
    lta NBINCOL $skpNBINCOL
    lta SKIPROW $skpSKIPROW
    lta SKIPCOL $skpSKIPCOL
    lta runseq
}

doCleanandClear() {
    source eraseANDepurge.sh
    lta set vsub $VSUB
    lta sseq $clearseq
    lta runseq
}

doSettings() {
    
    #unbinned
    # totNCOL=3600
    # totNROW=580
    

    skpNBINROW=1
    skpNBINCOL=1
    skpNCOL=500
    skpNROW=150
    vertOverscan=100
    horzOverscan=20

    let "totNCOL = 3080+ vertOverscan" 
    let "totNROW = 513+ horzOverscan"


    # skpNCOL=3600
    # skpNROW=580
    # skpSKIPCOL=0
    # skpSKIPROW=0
    let "skpSKIPCOL = totNCOL - skpNCOL"
    let "skpSKIPROW = totNROW - skpNROW"
    # generate uniform charge
    NSPURIOUSSHIFTS=200000 
    echo "running unbinned with $NSPURIOUSSHIFTS shifts"

    echo "total columns $totNCOL"
    echo "total rows $totNROW"
    echo "row binning $skpNBINROW"
    echo "column binning $skpNBINCOL"
    echo "image rows $skpNROW"
    echo "image columns $skpNCOL"
    echo "skip rows $skpSKIPROW"
    echo "skip columns $skpSKIPCOL"
    
}

doFullImage() {
    
    #unbinned
    totNCOL=3600
    totNROW=580
    

    skpNBINROW=1
    skpNBINCOL=1
    skpNCOL=3600
    skpNROW=580
    # vertOverscan=400
    # horzOverscan=50

    # let "totNCOL = 3080+ vertOverscan" 
    # let "totNROW = 513+ horzOverscan"


    # skpNCOL=3600
    # skpNROW=580
    skpSKIPCOL=0
    skpSKIPROW=0
    # let "skpSKIPCOL = totNCOL - skpNCOL"
    # let "skpSKIPROW = totNROW - skpNROW"
    # generate uniform charge
    NSPURIOUSSHIFTS=200000 
    echo "running unbinned with $NSPURIOUSSHIFTS shifts"

    echo "total columns $totNCOL"
    echo "total rows $totNROW"
    echo "row binning $skpNBINROW"
    echo "column binning $skpNBINCOL"
    echo "image rows $skpNROW"
    echo "image columns $skpNCOL"
    echo "skip rows $skpSKIPROW"
    echo "skip columns $skpSKIPCOL"
    
}


NSPURIOUSSHIFTS=200000
doFullImage
source eraseANDepurge.sh
lta set vsub $VSUB
lta sseq $clearseq
lta runseq
echo "erase and purge done"

lta sseq $imgseq
lta name $imgFOLDER/skp_${runname}_${skpTemp}k_binned_NROW${skpNROW}_NBINROW${skpNBINROW}_NCOL${skpNCOL}_NBINCOL${skpNBINCOL}_dummy_
doSkipper
echo "dummy image done"

for skpTemp in 180 183 187 190 193 197 200 203 207 
do 
    ../monitoring/slowControl/lakeshore_settemp.py $skpTemp
    sleep 12600
    source $spuriousvoltagescript
    vlow=$vl
    vhigh=$vh
    lta sseq $scseq
    #turn amplifier off
    lta set vdd_sw 0
    NSPURIOUSSHIFTS=200000
    lta NSPURIOUSSHIFTS $NSPURIOUSSHIFTS
    doRunseq
    echo "shifts done"
    lta set vdd_sw 1
    lta sseq $imgseq
    lta name $imgFOLDER/skp_${runname}_${skpTemp}k_binned_NROW${skpNROW}_NBINROW${skpNBINROW}_NCOL${skpNCOL}_NBINCOL${skpNBINCOL}_SC${NSPURIOUSSHIFTS}_vl${vlow}_vh${vhigh}_
    doSkipper
    echo "image done"
    for dtph in 750 1200 2000 3000 5000 8000 18000 28000 45000 70000 100000 170000 260000 400000 650000 1000000 1500000 1800000 2000000 2500000 3500000 6500000 8500000 10500000 15500000
    do

        #clear
        lta sseq $clearseq
        lta runseq
        

        source $spuriousvoltagescript
        vlow=$vl
        vhigh=$vh
        lta sseq $scseq
        #turn amplifier off
        lta set vdd_sw 0
        NSPURIOUSSHIFTS=200000
        lta NSPURIOUSSHIFTS $NSPURIOUSSHIFTS
        doRunseq
        echo "shifts done"


        #pocket pump
        source $highervoltagescript

        NPUMPS=3000
        dtph_long=$((dtph * 7))
        lta sseq $pumponlyseq
        #turn amplifier off
        lta set vdd_sw 0
        lta delay_Tph $dtph
        lta delay_Tph_long $dtph_long
        lta NPUMPS $NPUMPS
        lta runseq
        echo "Pumping complete"

        #turn amplifier back on
        lta set vdd_sw 1
        lta sseq $imgseq
        lta name $imgFOLDER/skp_${runname}_${skpTemp}k_binned_NROW${skpNROW}_NBINROW${skpNBINROW}_NCOL${skpNCOL}_NBINCOL${skpNBINCOL}_SC${NSPURIOUSSHIFTS}_vl${vlow}_vh${vhigh}_dtph${dtph}_NPUMPS${NPUMPS}_
        doSkipper
        echo "image done"

    done
done


cd $imgFOLDER
./skipper_all.sh
./cal_all.sh

cd ~/Soft/ltaDaemon


echo "done"|mailx -s "minos  dipole run done" -- $email


