OUTFILE=final_methods_llama7B_fp32.csv
PYFILE=profile_llama7B.py
python $PYFILE --ths -1.0 --filename $OUTFILE --cuths -1.0
python $PYFILE --ths 0.005 --filename $OUTFILE --cuths 0.005
python $PYFILE --ths 0.01 --filename $OUTFILE --cuths 0.01
python $PYFILE --ths 0.03 --filename $OUTFILE --cuths 0.03
python $PYFILE --ths 0.05 --filename $OUTFILE --cuths 0.05
python $PYFILE --ths 0.1 --filename $OUTFILE --cuths 0.1
python $PYFILE --ths 0.15 --filename $OUTFILE --cuths 0.15
python $PYFILE --ths 0.2 --filename $OUTFILE --cuths 0.2
python $PYFILE --ths 0.25 --filename $OUTFILE --cuths 0.25
python $PYFILE --ths 0.3 --filename $OUTFILE --cuths 0.3
python $PYFILE --ths 1.0 --filename $OUTFILE --cuths 1.0
python $PYFILE --ths 2.0 --filename $OUTFILE --cuths 2.0
python $PYFILE --ths 3.0 --filename $OUTFILE --cuths 3.0
python $PYFILE --ths 4.5 --filename $OUTFILE --cuths 4.5