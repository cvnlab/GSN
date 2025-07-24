#!/bin/sh
uids="cvn7009 cvn7012 cvn7002 cvn7011 cvn7007 cvn7006 cvn7013 cvn7016"
hemis="lh"
parcs="1 2 3"
parc_col="external_parc"
permute="False"
for uid in $uids ; do
  for hemi in $hemis ; do
    for parc in $parcs ; do
      echo "save_gsn_results.sh $uid $hemi $parc $parc_col $permute"
      sbatch save_gsn_results.sh $uid $hemi $parc $parc_col $permute
    done
  done
done
