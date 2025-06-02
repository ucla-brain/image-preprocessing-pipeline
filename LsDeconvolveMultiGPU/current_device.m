function device = current_device(gpu)
    device = 'CPU';
    if gpu > 0
        device = sprintf('GPU%d', gpu);
    end
end