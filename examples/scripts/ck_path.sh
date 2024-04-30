check_symbolic_link() {
    local path="$1"
    local origin="$2"

    if [ ! -e "$path" ]; then
        ln -s "$origin" "$path"
        echo "Soft link created at $path"
    elif [ ! -L "$path" ]; then
        echo "Please remove $path and create a soft link to $origin by:"
        echo "\$ ln -s $origin $path"
        exit 1
    fi
}

if [ "$(hostname)" = "dmal-triangle-001" ]; then
    check_symbolic_link "../log"  "/share/data/transfer/Spectral-GNN-Benchmark/log"
    check_symbolic_link "../data" "/share/data/dataset/PyG/"
fi
