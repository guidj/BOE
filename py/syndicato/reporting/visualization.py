import os
import os.path

import syndicato
from syndicato.reporting import plotting


def imports():
    return {
        'https://fonts.googleapis.com/css?family=Karla': 'stylesheet'
    }


def load_asset(name):
    file_path = os.path.join(syndicato.BASE_DIR, name)

    with open(file_path, 'r') as fp:
        return fp.read()


def html_report(arms, results, output_path):
    title = 'Bandit Monte Carlo Simulation'
    from yattag import Doc

    doc, tag, text = Doc().tagtext()

    with tag('head'):
        with tag('title'):
            text(title)

        for k, v in imports().items():
            with tag('link', href=k, ref=v):
                pass

        with tag('style', type='text/css'):
            text(load_asset('assets/styling.css'))

    with tag('body'):
        with tag('h1'):
            text(title)

        with tag('h2'):
            text('Arms')

        for arm in arms:
            with tag('p'):
                text(str(arm))

        for exp_config, plots in results:
            with tag('div'):
                with tag('h2'):
                    text('Experiment Configuration')

                for param in sorted(exp_config.keys()):
                    with tag('p'):
                        text('{}: {}'.format(param, exp_config[param]))

                with tag('h2'):
                    text('Results')

                with tag('div', klass='card card-3'):
                    for plot in plots:
                        encoded_image = plotting.encode_figure_as_base64(plot)
                        with tag('div', klass='metrics-plot'):
                            doc.stag('img', src='data:image/png;base64, {}'.format(encoded_image.decode('utf-8')),
                                     klass="photo")

    with open(output_path, 'w') as fp:
        fp.write(doc.getvalue())
