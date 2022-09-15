#!/bin/sh

for GPU_COUNT in 1; do 
	datasets=$(ls ../datasets/test)
	for file in $datasets; do 
		for eig_count in 8 12 16 20 24; do 
			echo "Doing datasets/test/$file with eig_count=$eig_count on $GPU_COUNT gpu -> float double float";
			node --polyglot --vm.Dtruffle.class.path.append=$GRAAL_HOME/languages/grcuda/grcuda.jar --experimental-options --grcuda.ExecutionPolicy=async --grcuda.ForceStreamAttach --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.DependencyPolicy=with-const --grcuda.RetrieveParentStreamPolicy=disjoint --grcuda.InputPrefetch --grcuda.MemAdvisePolicy=none --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.NumberOfGPUs=$GPU_COUNT --grcuda.DeviceSelectionPolicy=min-transfer-size main.js ../datasets/test/$file $eig_count 10 $GPU_COUNT true false float double float >> results_orth_float_double_float_c1
			echo "Doing datasets/test/$file with eig_count=$eig_count on $GPU_COUNT gpu -> double double double";
			node --polyglot --vm.Dtruffle.class.path.append=$GRAAL_HOME/languages/grcuda/grcuda.jar --experimental-options --grcuda.ExecutionPolicy=async --grcuda.ForceStreamAttach --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.DependencyPolicy=with-const --grcuda.RetrieveParentStreamPolicy=disjoint --grcuda.InputPrefetch --grcuda.MemAdvisePolicy=none --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.NumberOfGPUs=$GPU_COUNT --grcuda.DeviceSelectionPolicy=min-transfer-size main.js ../datasets/test/$file $eig_count 10 $GPU_COUNT true false double double double >> results_orth_double_double_double_c1
		done;
	done;
done;
