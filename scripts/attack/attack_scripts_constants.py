checkpoint_strategies2checkpoint_strings_llama_secalign = {
    "step": """checkpoints1=$(seq 0 30)
checkpoints2=$(seq 50 50 898)
checkpoints3=897
checkpoints=($checkpoints1 $checkpoints2 $checkpoints3)
echo "Checkpoints: ${checkpoints[@]}"
    """,
    "loss": """checkpoints=(0   1   3   4   5   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  24  25  26  27  29  31  32  33  36  37  38  40  41  45  46  49  50  59  60  61  65  66  72  74  77  78  85  91  92  95  96  98  99 102 103 104 109 111 118 121 125 126 131 134 152 153 154 159 160 162 164 165 166 171 172 174 175 177 178 189 190 191 192 200 201 219 220 222 223 224 225 226 238 239 267 268 274 275 277 278 280 281 285 286 336 337 338 388 432 433 456 457 507 557 607 657 707 757 776 777 827 877 897)
echo "Checkpoints: ${checkpoints[@]}"
    """,
    "gradnorm": """checkpoints=(0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  37  38  40  41  43  45  46  49  54  58  59  60  65  72  77  78  79  80  81  84  91  98 102 103 104 109 110 117 118 119 120 125 152 153 157 159 162 163 165 171 174 177 186 189 191 200 219 222 223 225 235 238 257 265 267 274 277 280 285 287 382 432 456 464 484 897)
echo "Checkpoints: ${checkpoints[@]}"
""",
}

checkpoint_strategies2checkpoint_strings_mistral_secalign = {
    "step": """checkpoints1=$(seq 0 30)
checkpoints2=$(seq 50 50 898)
checkpoints3=897
checkpoints=($checkpoints1 $checkpoints2 $checkpoints3)
echo "Checkpoints: ${checkpoints[@]}"
    """,
    "gradnorm": """checkpoints=(0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19   20   21   22   24   25   27   28   29   30   31   32   33   34   36   37   38   39   44   48   50   63   68   69   76   78   84   90   91   106   109   111   124   133   136   139   155   164   167   171   173   174   188   198   201   215   222   235   236   242   250   253   265   266   267   290   299   303   304   329   529   539   540   555   559   580   585   588   594   638   818   832   866   897)
echo "Checkpoints: ${checkpoints[@]}"
""",
}
checkpoint_strategies2checkpoint_strings_llama_struq = {
    "gradnorm": """checkpoints=(0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   30   31   33   35   36   41   42   43   47   50   63   96  100  104  180  197  220  281  282  287  325  379  407  437  479  593  652  695  711  727  777  842  963 1004 1017 1045 1064 1109 1152 1165 1218 1281 1377 1504 1546 1607 1662 1665 1688 1701 1705 1730 1731 1739 1739 1757 1778 1793 1830 1849 1859 1872 1887 1901 1910 1938 1953 1962 1964 1997 2000 2047 2055 2065 2084 2087 2102 2104 2117 2119 2135 2139 2153 2156 2177 2196 2209 2227 2251 2286 2299 2301 2319 2320 2356 2360 2371 2421 2424)
echo "Checkpoints: ${checkpoints[@]}"
"""
}

checkpoint_strategies2checkpoint_strings_mistral_struq = {
    "gradnorm": """checkpoints=(0 1 2 3 4 5 6 7 8 9 11 13 20 24 30 35 43 47 50 87 94 111 158 180 189 194 221 236 247 268 274 281 290 300 306 316 319 325 378 398 400 403 407 413 418 435 465 471 474 476 499 501 521 526 537 539 563 582 593 624 636 652 687 689 690 731 741 762 765 777 807 826 842 853 886 887 1015 1017 1029 1064 1101 1109 1165 1204 1210 1220 1252 1281 1314 1320 1372 1388 1393 1443 1452 1456 1474 1522 1583 1591 1601 1612 1690 1695 1731 1765 1778 1836 1962 1964 2418 2424)
echo "Checkpoints: ${checkpoints[@]}"
"""
}

checkpoint_strategies2checkpoint_strings_safety_llama = {
    "gradnorm": """checkpoints=(0   1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35   36   37   38   39   40   41   42   43   44   45   46   47   51   53   55   66   75   77   84   85   100   118   140   145   168   197   229   235   263   265   268   270   282   300   303   308   309   322   325   330   334   335   340   345   356   359   365   366   367   370   371   382   383   385   387   389   393   395   400   401   402   410   411   419   426   436   446   448   451   453   455   459   463   464   467   471   474   482   486   494   495   498   501   502   503   505   506   513   514   516   517   518   519   524   525   526   527   528   531   532   539   541   542   543   546   548   552   554   555   558   559   562   564   568   569   573   576   577   578   579   585   587   589   592   593   594   596   598   600   603   605   606   610   611   617   619   620   621   622   627   629   630   631   632   633   636   637   639   640   641   642   645   646   649   650   652   654   656   657   659   660   661   662   663   665   666   668)
echo "Checkpoints: ${checkpoints[@]}"
"""
}

hpc_header = """#PBS -l walltime=23:59:59
#PBS -l select=1:ncpus=1:mem=30gb:ngpus=1

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate secalign
cd /rds/general/user/xy2222/ephemeral/attack-secalign-fix-mistral
"""

checkpoint_attack_bash_script = """
current_time="{current_time}"
echo "Current time: $current_time"

# Description:{description}

for checkpoint in "${{checkpoints[@]}}"
do
    python test.py \\
        --model_name_or_path "{model_path}" \\
        --device "0" \\
        --defense "{defense}" \\
        --data_path "{data_path}" \\
        --checkpoint_dir "/checkpoint_gcg/" \\
        --checkpoint $checkpoint \\
        --all_checkpoints ${{checkpoints[@]}} \\
        --checkpoint_choice "{checkpoint_choice}" \\
        --current_time $current_time \\
        --sample_ids {sample_ids} \\
        --gcg_global_budget \\
        --gcg_early_stopping \\
        {universal_or_individual_attack_args}
        {additional_args}
done
"""

direct_attack_bash_script = """
current_time="{current_time}"
echo "Current time: $current_time"

# Description:
{description}

python test.py \\
    --model_name_or_path "{model_path}" \\
    --device "0" \\
    --defense "{defense}" \\
    --data_path "{data_path}" \\
    --checkpoint_dir "/checkpoint_gcg/" \\
    --current_time $current_time \\
    --sample_ids {sample_ids} \\
    --gcg_global_budget \\
    --gcg_early_stopping \\
    {universal_or_individual_attack_args} 
    {additional_args}
"""

individual_sample_attack_args = (
    """--gcg_num_steps_per_checkpoint 1000 --gcg_num_steps_total 50000"""
)

universal_attack_args = """--gcg_num_steps_per_sample 3500 --gcg_num_steps_total 10000 --gcg_universal_attack"""
