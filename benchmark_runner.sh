# to run this 

#chmod +x benchmark_runner.sh
#./benchmark_runner.sh


if ! command -v hyperfine &> /dev/null; then
    echo "Hyperfine not found. Installing..."
    sudo apt update
    sudo apt install -y hyperfine
else
    echo "Hyperfine is already installed."
fi

if [ ! -f "arg.txt" ]; then
    echo "arg.txt not found! Please make sure the file exists in the current directory."
    exit 1
fi

command_to_run=$(<arg.txt)

echo "Testing command before benchmarking..."
eval "$command_to_run" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Error: Command failed. Not running benchmark or writing result file."
    exit 1
fi

base_name="result"
ext="txt"
filename="${base_name}.${ext}"
counter=1

while [ -f "$filename" ]; do
    filename="${base_name}_${counter}.${ext}"
    ((counter++))
done

echo "Running benchmark on: $command_to_run"
hyperfine --runs 10 "$command_to_run" --export-markdown "$filename"

echo "✅ Benchmark complete. Results saved to $filename."
