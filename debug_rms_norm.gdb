# GDB script to analyze RMS norm calls
set breakpoint pending on
set print pretty on

# Set breakpoint in our fake quantization function
break should_apply_fake_quant_rms_norm

# Commands to run when breakpoint hits
commands
  printf "=== RMS NORM CALL ===\n"
  printf "Tensor name: %s\n", tensor->name ? tensor->name : "NULL"
  printf "Thread ID: %ld\n", syscall(186)  # gettid
  
  # Print stack trace (first 3 frames)
  printf "Call stack:\n"
  bt 3
  
  printf "Global call count: %d\n", g_global_call_count
  printf "==================\n"
  
  # Continue execution
  continue
end

# Run with limited execution time
run -m "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" -p "Hi" -n 0 --fake-quant-ffn-norm bf16 --fake-quant-layer 5 --no-warmup