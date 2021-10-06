function run_repeats()
    for seed in 1:100
        # create .sh file to submit jobs
        filename = "sim$seed.sh"
        open(filename, "w") do io
            println(io, "#!/bin/bash")
            println(io, "#")
            println(io, "#SBATCH --job-name=sim$seed")
            println(io, "#")
            println(io, "#SBATCH --time=24:00:00")
            println(io, "#SBATCH --cpus-per-task=16")
            println(io, "#SBATCH --mem-per-cpu=8G")
            println(io, "#SBATCH --partition=owners")
            println(io, "")
            println(io, "#save job info on joblog:")
            println(io, "echo \"Job \$JOB_ID started on:   \" `hostname -s`")
            println(io, "echo \"Job \$JOB_ID started on:   \" `date `")
            println(io, "")
            println(io, "# load the job environment:")
            println(io, "module load julia")
            println(io, "")
            println(io, "# run code")
            println(io, "export JULIA_NUM_THREADS=16")
            println(io, "echo 'julia prs.jl $seed'")
            println(io, "julia prs.jl $seed")
            println(io, "")
            println(io, "#echo job info on joblog:")
            println(io, "echo \"Job \$JOB_ID ended on:   \" `hostname -s`")
            println(io, "echo \"Job \$JOB_ID ended on:   \" `date `")
            println(io, "#echo \" \"")
        end
        
        # submit job
        run(`sbatch $filename`)
        println("submitted job $seed")
        rm(filename, force=true)
        sleep(1)
    end
end
run_repeats()
