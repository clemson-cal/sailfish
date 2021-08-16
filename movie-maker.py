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
        mesh = chkpt['mesh']
        prim = np.zeros([mesh['ni'], mesh['nj'], 3])
        for patch in chkpt['primitive_patches']:
            i0 = patch['rect'][0]['start']
            j0 = patch['rect'][1]['start']
            i1 = patch['rect'][0]['end']
            j1 = patch['rect'][1]['end']
            local_prim = np.array(np.frombuffer(patch['data'])).reshape([i1 - i0, j1 - j0, 3])
            prim[i0:i1, j0:j1] = local_prim
        plt.imshow(prim[:,:,0].T, origin='lower')
        plt.title(r"{} $\Sigma^{{1/4}}$".format(name))

        file_count_str = str(file_count)

        if len(file_count_str) < max_file_count:
            file_count_str = ('0' * (max_file_count - len(file_count_str))) + file_count_str

        fname = '{}/output-figures/movie-{}.png'.format(current_path_name, file_count_str)
        print(fname)
        plt.savefig(fname, dpi=600)

    make_movie(current_path_name, outdir, filename, max_file_count)

    if savefigbool is False:
        os.system("rm -rf {}/{}".format(current_path_name, 'output-figures'))


def make_movie(current_path, outdir, filename, max_count):
    Path('{}/{}'.format(current_path, outdir)).mkdir(parents=True, exist_ok=True)
    command = "ffmpeg -start_number 1 -i {}/output-figures/movie-%0{}d.png -c:v libx264 -vb 20M -r 30 -pix_fmt yuv420p -filter:v 'setpts=2*PTS' -y {}/movie-{}.mp4".format(current_path, max_count, outdir, filename)

    os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='', help='Checkpoint file directory.', required=True)
    parser.add_argument('--outdir', default='movie', help='Output movie directory.')
    parser.add_argument('--filename', default='movie', help='Output movie name.')
    parser.add_argument('--savefigs', default=False, help='Whether the program saves the figures used to make the movie.')
    args = parser.parse_args()

    file_load(args.indir, args.outdir, args.savefigs, args.filename)
