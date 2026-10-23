#!/bin/bash
set -e

function watch_refit {
for h in `seq 0 1 5000`; do
      if squeue --format="%j" | grep -q "refit_$specname"; then
              echo 'still refiting, keep watching'
              sleep 120
      else
              break
      fi
done
}

function watch_dft {
for h in `seq 0 1 5000`; do
      if squeue --format="%j" | grep -q "dft_$specname"; then
              echo 'still dfting, keep watching'
              sleep 120
      else
              break
      fi
done
}

function watch_select {
for h in `seq 0 1 5000`; do
      if squeue --format="%j" | grep -q "select_$specname"; then
              echo 'still selecting, keep watching'
              sleep 60
      else
              break
      fi
done
}

function watch_relax {
for h in `seq 0 1 5000`; do
      if squeue --format="%j" | grep -q "relax_$specname"; then
              echo 'still relaxing, keep watching'
              sleep 30
      else
              break
      fi
done
}

function watch_sample {
for h in `seq 0 1 5000`; do
      if squeue --format="%j" | grep -q "sample_$specname"; then
              echo 'still sampling, keep watching'
              sleep 120
      else
              break
      fi
done
}

function watch_upgrade {
for h in `seq 0 1 5000`; do
      if squeue --format="%j" | grep -q "upgrade_$specname"; then
              echo 'still upgrading, keep watching'
              sleep 30
      else
              break
      fi
done
}

function watch_filter {
for h in `seq 0 1 5000`; do
      if squeue --format="%j" | grep -q "filter_$specname"; then
              echo 'still filtering, keep watching'
              sleep 30
      else
              break
      fi
done
}

function watch_split {
for h in `seq 0 1 5000`; do
      if squeue --format="%j" | grep -q "split_$specname"; then
              echo 'still spliting, keep watching'
              sleep 30
      else
              break
      fi
done
}

function watch_convert {
for h in `seq 0 1 5000`; do
      if squeue --format="%j" | grep -q "convert_$specname"; then
              echo 'still converting, keep watching'
              sleep 30
      else
              break
      fi
done
}

function watch_reduce_size {
for h in `seq 0 1 5000`; do
      if squeue --format="%j" | grep -q "reduce_size_$specname"; then
              echo 'still reducing, keep watching'
              sleep 30
      else
              break
      fi
done
}

outcwd=$(pwd)
for s in `seq 1 1 1`; do
	cd $outcwd
	mkdir str_$s
	cd str_$s
	cp -r ../ori_source/* ./
	cp ../data_source/data$s* ./source/data1.pos
	specname=$RANDOM
	cwd=$(pwd)

	rm -r temp || true
	mkdir temp
	rm -r fitted_mtps || true
	mkdir fitted_mtps
	cp bulk_mtps/6.cfg temp/selectedcfgs.temp

	for l in `seq 6 2 18`; do
		date
		cd $cwd
		cp bulk_mtps/$l.mtp temp/mtp.temp
		mkdir al_$l
		cd al_$l

		for p in `seq 0 1 0`; do
        		mkdir pre$p
        		cd pre$p
        		cp $cwd/temp/mtp.temp ./init.mtp
			if [[ "$p" -eq 0 ]]; then	
				cp $cwd/temp/selectedcfgs.temp ./selectedcfgs.cfg	
			else
				cat $cwd/temp/selectedcfgs.temp $cwd/temp/diff.temp > ./selectedcfgs.cfg
			fi
			sed -e 's|xxxNAMExxx|'$specname'|' $cwd/bulk_mtps/$l.job > ./job
        		sbatch job
			watch_refit
			#sed -e 's|xxxCWDxxx|'$cwd'|' -e 's|xxxNAMExxx|'$specname'|' $cwd/source/pre_select.job > ./select.job
			#sbatch select.job
			#watch_select	
        		cp mlip.mtp $cwd/temp/mtp.temp
        		#cp diff.cfg $cwd/temp/diff.temp
        		#cp selectedcfgs.cfg $cwd/temp/selectedcfgs.temp
			cd ..
		done
	
		for cfg in `seq 1 1 1`; do
			cd ${cwd}/al_${l}
		for i in `seq 1 1 100`; do
			cd ${cwd}/al_${l}
			mkdir cfg${cfg}_${i}
			cd cfg${cfg}_${i}

			# new sample
			for q in {1,2,3,4}; do
				mkdir sample$q
				cd sample$q
				cp $cwd/temp/mtp.temp ./mlip.mtp
				cp $cwd/source/data$cfg.pos ./data.pos
				cp $cwd/source/lammps_input ./
				sed -e 's|xxxNAMExxx|'$specname'|' $cwd/source/sample.job > ./job
				sbatch job
				cd ..
			done
			watch_sample
		
			for s in sample*; do
				cd $s
				cp $cwd/source/reduce_size.sh ./
				sed -e 's|xxxNAMExxx|'$specname'|' $cwd/source/reduce_size.job > ./reduce_size.job
				sbatch reduce_size.job
				watch_reduce_size
				
				sed -e 's|xxxCWDxxx|'$cwd'|' -e 's|xxxNAMExxx|'$specname'|' $cwd/source/al_select.job > ./select.job
				sbatch select.job
				cd ..
			done
			watch_select

			cat sample1/sampledcfgs >> $cwd/temp/sampledcfgs || true
			cat sample2/sampledcfgs >> $cwd/temp/sampledcfgs || true
			cat sample3/sampledcfgs >> $cwd/temp/sampledcfgs || true
			cat sample4/sampledcfgs >> $cwd/temp/sampledcfgs || true

			mkdir extract
			cd extract
			cp ../sample1/mlip.mtp ./
			cat ../sample1/sampledcfgs_add >> sampledcfgs || true
			cat ../sample2/sampledcfgs_add >> sampledcfgs || true
			cat ../sample3/sampledcfgs_add >> sampledcfgs || true
			cat ../sample4/sampledcfgs_add >> sampledcfgs || true
			rm -f ../sample1/sampledcfgs* || true
                	rm -f ../sample2/sampledcfgs* || true
                	rm -f ../sample3/sampledcfgs* || true
                	rm -f ../sample4/sampledcfgs* || true
			sed -e 's|xxxCWDxxx|'$cwd'|' -e 's|xxxNAMExxx|'$specname'|'  $cwd/source/al_select.job > ./select.job
			sbatch select.job
                	watch_select
		
			#check if sampledcfgs exists, if not break the for loop
			nextlevel=1
			selectinfo=$(mlp3 cut_extrapolative_nbh sampledcfgs_add sampled_nbh --cutoff=20)
			if [ -z "${selectinfo}" ]; then
				echo 'no cfg is smapled, finish active learning'
				cp ../sample1/mlip.mtp $cwd/fitted_mtps/$l.mtp
				mlp3 check_errors ../sample1/mlip.mtp $cwd/temp/selectedcfgs.temp --report_to=$cwd/fitted_mtps/$l.out
				break
			fi

			echo "$selectinfo"
			gradelist=( $(echo "$selectinfo" | cut -d' ' -f 6 | grep -Eo '[0-9.e+]+$') )
			for grade in "${gradelist[@]}" ; do
				gradenum=( $(sed -E 's/([+-]?[0-9.]+)[eE]\+?(-?)([0-9]+)/(\1*10^\2\3)/g' <<<"$grade"))
				if (( $(echo "$gradenum > 1.7" |bc -l) )); then
					nextlevel=0
				fi
			done
			if (( nextlevel )); then
				echo 'grades of all cfgs are smaller than 1.7, finish active learning'
				cp ../sample1/mlip.mtp $cwd/fitted_mtps/$l.mtp
				mlp3 check_errors ../sample1/mlip.mtp $cwd/temp/selectedcfgs.temp --report_to=$cwd/fitted_mtps/$l.out
				break
			fi

			mlp3 convert --input_format=txt --output_format=lammps_datafile sampled_nbh data
		
			cp $cwd/source/cut2box.py ./
			cut_idx=0
			cut_idx_out=0
			count=$(ls -1q ./data* | wc -l)
			if [[ "$count" -eq 1 ]]; then
				python cut2box.py data ${cut_idx}
			else
				for f in ./data*; do
					python cut2box.py data${cut_idx} ${cut_idx_out}
					((cut_idx+=1))
					if compgen -G "cutdata${cut_idx_out}" ; then
						((cut_idx_out+=1))
					fi
				done
			fi
			cd ..
		
			# relax local structure
			mkdir relax
			cd relax
			mkdir poscar
			re_idx=0
			for f in ../extract/cutdata*; do
				mkdir re$re_idx
				cp $cwd/source/lammps_input_relax ./re$re_idx
				sed -e 's|xxxNAMExxx|'$specname'|' $cwd/source/relax.job > ./re$re_idx/relax.job
				cp ../extract/cutdata${re_idx} ./re$re_idx/cutdata.pos
				cd re$re_idx
				sbatch relax.job
				cd ..
				((re_idx+=1))
                	done
			watch_relax
		
			idx=0
			for d in re*; do	
                        	cd ${d}
				if compgen -G "cutdata.out" ; then
                        		cp $cwd/source/data2poscar.py ./
					python data2poscar.py cutdata.out
					if compgen -G "POSCAR_*" ; then
						cp POSCAR_* ../poscar/POSCAR$idx
						((idx+=1))
					fi
				fi
				cd ..
                	done
                	cd ..
		
			# new dft
			mkdir low_dft
			cd low_dft
			cp -r ../relax/poscar ./
			cp -r $cwd/source/low_dft ./source
			sed -e 's|xxxNAMExxx|'$specname'|' $cwd/source/low_dft/job > ./source/job
			count=$(ls -1q ./poscar/ | wc -l)
			if ! [[ "$count" -eq 0 ]]; then
				for j in `seq 0 1 $(($count-1))`; do
					mkdir cal$j
        				cp source/* ./cal$j
        				cp poscar/POSCAR$j ./cal$j/POSCAR
        				cd cal$j
        				sbatch job
        				cd ..
				done
				watch_dft
			
				cd ..
				mkdir high_dft
				cd high_dft
				cp -r $cwd/source/high_dft ./source
				sed -e 's|xxxNAMExxx|'$specname'|' $cwd/source/high_dft/job > ./source/job
				for j in `seq 0 1 $(($count-1))`; do
					mkdir cal$j
					cp source/* ./cal$j
					cp ../low_dft/cal$j/CONTCAR ./cal$j/POSCAR
					cd cal$j
					#python removeconstrain.py
					sbatch job
					cd ..
				done
				watch_dft

				sed -e 's|xxxNAMExxx|'$specname'|' $cwd/source/convert.job > ./convert.job
				sbatch convert.job
				watch_convert
				cd ..

				# new refit
				mkdir refit
				cd refit
				cat $cwd/temp/selectedcfgs.temp ../high_dft/diff.cfg > ./selectedcfgs.cfg
				cp ../sample1/mlip.mtp ./init.mtp
				sed -e 's|xxxNAMExxx|'$specname'|' $cwd/bulk_mtps/$l.job > ./job
				sbatch job
				watch_refit
                		cp mlip.mtp $cwd/temp/mtp.temp
                		cp selectedcfgs.cfg $cwd/temp/selectedcfgs.temp

				#index+filter
				vim -es -c 'let n=1 | g/Feature   ind/exe "normal o Feature   idx " . n | let n=n+1' -c 'wq' $cwd/temp/sampledcfgs
				sed -e 's|xxxNAMExxx|'$specname'|' -e 's|xxxCWDxxx|'$cwd'|' -e 's|xxxCWDxxx|'$cwd'|' $cwd/source/filter_grade.job > ./filter_grade.job
				sbatch filter_grade.job
                        	watch_filter
                        	if compgen -G "$cwd/temp/sampledcfgs_filtergrade.0" ; then
					mlp3 mindist_filter $cwd/temp/sampledcfgs_filtergrade.0 1.6 $cwd/temp/sampledcfgs_filterall
					rm -f $cwd/temp/sampledcfgs_filtergrade.0 || true
				
					if [ -s "$cwd/temp/sampledcfgs_filterall" ]; then
						#split
						cd $cwd/temp
						mkdir process_cfgs
						cd process_cfgs
						sed -e 's|xxxINPUTxxx|'../sampledcfgs_filterall'|' $cwd/source/split.sh > ./split.sh
                        			chmod +x split.sh
						sed -e 's|xxxNAMExxx|'$specname'|' $cwd/source/split.job > ./split.job
                        			sbatch split.job
                        			watch_split

						#combine
						cp $cwd/source/upgrade.sh ./
						sed -e 's|xxxNAMExxx|'$specname'|' $cwd/source/upgrade.job > ./upgrade.job
						sbatch upgrade.job
						watch_upgrade
						cd ..
					fi
				fi

				rm -f $cwd/temp/sampledcfgs || true
				rm -f $cwd/temp/sampledcfgs_filterall || true
				rm -r -f $cwd/temp/process_cfgs || true
				if compgen -G "$cwd/temp/sampledcfgs_new" ; then
					sed -i '/Feature\s\+idx/d' $cwd/temp/sampledcfgs_new
					mv $cwd/temp/sampledcfgs_new $cwd/temp/sampledcfgs
				fi
				cd ..
			fi

		done
		done
	date
	done
	cd ..
done
