import numpy as np
import matplotlib.pyplot as plt
import msgpack
import os
from pathlib import Path
import argparse


plt.switch_backend('agg')


def file_load(indir, outdir, savefigbool, filename):
    file_count = 0
    current_path_name = Path().resolve()
    Path('{}/output-figures'.format(current_path_name)).mkdir(parents=True, exist_ok=True)
    max_file_count = 5  # Number of digits in the filename.

    for name in Path(indir).iterdir():
        file_count += 1
        chkpt = msgpack.load(open(name, 'rb'))
        ni = chkpt['mesh']['ni']
        nj = chkpt['mesh']['nj']
        x0 = chkpt['mesh']['x0']
        y0 = chkpt['mesh']['y0']
        x1 = chkpt['mesh']['dx'] * chkpt['mesh']['ni'] + x0
        y1 = chkpt['mesh']['dy'] * chkpt['mesh']['nj'] + y0
        primitive = np.reshape(chkpt['primitive'], (ni + 4, nj + 4, 3))[2:-2,
                    2:-2]
        plt.figure(figsize=[12, 9.5])
        plt.imshow(primitive[:, :, 0].T ** 0.25, origin='lower', cmap='plasma',
                   extent=[x0, x1, y0, y1])
        plt.colorbar()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.title(r"{} $\Sigma^{{1/4}}$".format(name))

        file_count_str = str(file_count)

        if len(file_count_str) < max_file_count:
            file_count_str = ('0' * (max_file_count - len(file_count_str))) + file_count_str

        fname = '{}/output-figures/movie-{}.png'.format(current_path_name, file_count_str)
        print(fname)
        plt.savefig(fname)

    make_movie(current_path_name, outdir, filename, max_file_count)

    if savefigbool is False:
        os.system("rm -rf {}/{}".format(current_path_name, 'output-figures'))


def make_movie(current_path, outdir, filename, max_count):
    Path('{}/{}'.format(current_path, outdir)).mkdir(parents=True, exist_ok=True)
    # command = "ffmpeg -start_number 1 -f image2 -r 24 -i {}/output-figures/%0{}d.png -vcodec mpeg4 -y {}/movie-{}.mp4".format(current_path, max_count, outdir, filename)

    command = "ffmpeg -start_number 1 -i {}/output-figures/%0{}d.png -c:v libx264 -vb 20M -r 30 -pix_fmt yuv420p -filter:v 'setpts=2*PTS' -y {}/movie-{}.mp4".format(current_path, max_count, outdir, filename)

    os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='', help='Checkpoint file directory.', required=True)
    parser.add_argument('--outdir', default='movie', help='Output movie directory.')
    parser.add_argument('--filename', default='movie', help='Output movie name.')
    parser.add_argument('--savefigs', default=False, help='Whether the program saves the figures used to make the movie.')
    args = parser.parse_args()

    file_load(args.indir, args.outdir, args.savefigs, args.filename)
