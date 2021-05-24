#!/bin/bash
F_FILE=tasks/lm/f_score_ptb.pkl
TF_FILE=tasks/lm/tf_score_ptb.pkl

# frequency-based score
if [ -f "$F_FILE" ]; then
    echo "$F_FILE exists."
else 
    echo "$F_FILE does not exist."
		echo "Computing now..."
		pushd tasks/lm
		python get_tfidf.py --data ptb --mode frequency
		pushd ../..
fi

# tfidf-based score
if [ -f "$TF_FILE" ]; then
    echo "$TF_FILE exists."
else 
    echo "$TF_FILE does not exist."
		echo "Computing now..."
		pushd tasks/lm
		python get_tfidf.py --data ptb --mode tfidf
		pushd ../..
fi

function do_test() {
	OPT=$1
	pushd ./experiment/lm/"$OPT"
	ln -s ../../../tasks/lm/.data
	if [ -f "baseline.p7" ]; then
		echo "Already done - $OPT"
	else 
		python -u ../../../run_lm.py --config base.yaml > output_log
	fi
	pushd ../../..	
}

echo "$1"

# Declare an array of string with type
declare -a StringArray=("base" "frequency" "tfidf" "frequency_refined" "tfidf_refined" "frequency_50" "tfidf_50", "smallfry", "smallfry_50", "word2ket", "word2ket_50")
  
# Iterate the string array using for loop
for val in ${StringArray[@]}; do
    echo $val
	if [ $1 == $val ] || [ $1 == "all" ]; then
		do_test $val
	fi
done
