#!/bin/sh

for GPU_COUNT in 2; do 
	datasets=$(ls /home/ubuntu/topk_eigen_dump/topk_eigen_dump/datasets)
	for file in $datasets; do 
		for eig_count in 8; do
			for policy in "minmax-transfer-time"; do 
				echo "Doing $file with eig_count=$eig_count on $GPU_COUNT gpu with $policy policy -> float float float";
				node --polyglot --vm.Dtruffle.class.path.append=$GRAAL_HOME/languages/grcuda/grcuda.jar --experimental-options --log.grcuda.com.nvidia.grcuda.runtime.level=FINE --grcuda.ExecutionPolicy=async --grcuda.ForceStreamAttach --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.DependencyPolicy=with-const --grcuda.RetrieveParentStreamPolicy=disjoint --grcuda.InputPrefetch --grcuda.MemAdvisePolicy=none --grcuda.RetrieveNewStreamPolicy=always-new --grcuda.NumberOfGPUs=$GPU_COUNT --grcuda.DeviceSelectionPolicy=$policy main.js /home/ubuntu/topk_eigen_dump/topk_eigen_dump/datasets/$file $eig_count 1 $GPU_COUNT
			done;
		done;
	done;
done;
