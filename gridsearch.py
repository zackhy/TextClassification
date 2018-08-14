__author__ = 'Duc'

# This program automate running the experiments (doing gridsearch).

import re
import subprocess
import fire

# network_types = ['cnn', 'blstm', 'clstm']
# hidden_layer_sizes = [256, 512]

models = {'cnn': ['', []],
          'blstm': ['--hidden_size=300', [300]],
          'clstm': ['', []]
         }

def extract_scores_and_run_dir(result_str):
    result_lines = result_str.split('\n')[-2:]
    m = re.match('Best validation: (0.\d+) \| (0.\d+) \| (0.\d+), at step: (\d+)', result_lines[0])
    prec, rec, f1, best_val_step = m.group(1), m.group(2), m.group(3), m.group(4)
    m = re.match('All the files have been saved to ([/\w-]+)', result_lines[1])
    out_dir = m.group(1)
    return prec, rec, f1, best_val_step, out_dir

def extract_test_scores(result_str):
    result_line = result_str.split('\n')[-1]
    m = re.match('Test scores: (0.\d+) \| (0.\d+) \| (0.\d+)', result_line)
    if not m:
        print('Cannot extract test scores, got the following output:')
        print(result_str.split('\n'))
    prec, rec, f1 = m.group(1), m.group(2), m.group(3)
    return prec, rec, f1

def run_train(model_name, model_cmd_args, train_data_file):
    """
    model_name: name of the neural net architecture.
    model_cmd_args: list of model-specific cmds."""
    cmd = 'python -m zackhy.train --data_file={} --save_every_steps=100 --max_steps=1500 --num_checkpoint=25 --clf={}'.format(train_data_file, model_name)
    train_cmd = cmd.split()
    train_cmd.extend(model_cmd_args)
    
    print('Model: {}'.format(model_name))
    print(cmd)
    result_str = subprocess.run(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode('utf-8')
    return extract_scores_and_run_dir(result_str.strip())

def run_test(best_val_step, out_dir, test_data_file):
    cmd = 'python -m zackhy.test --test_data_file={} --checkpoint=clf-{} --run_dir={}'.format(test_data_file, best_val_step, out_dir)
    print(cmd)
    result_str = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode('utf-8')
    return extract_test_scores(result_str.strip())

def gridsearch(train_data_file, test_data_file):
    for model_name, hyperparams in models.items():
        model_cmd_args = hyperparams[0]
        val_prec, val_rec, val_f1, best_val_step, out_dir = run_train(model_name, model_cmd_args, train_data_file)
        print('Best val step: {}'.format(best_val_step))
        print('Output dir: {}'.format(out_dir))
        print('Best validation scores: {} | {} | {}'.format(val_prec, val_rec, val_f1))

        test_prec, test_rec, test_f1 = run_test(best_val_step, out_dir, test_data_file)
        print('Test scores: {} | {} | {}'.format(test_prec, test_rec, test_f1))


if __name__ == '__main__':
    fire.Fire(gridsearch)
