import os
import sys
import time
import yaml
import logging

from spacy_api import client, server

# Globals
# LOG = logging.getLogger(__name__)
# LOG.setLevel(logging.DEBUG)
# FORMATTER = logging.Formatter('%(asctime)s - %(name)-8s - %(levelname)-8s:  %(message)s')
# CH = logging.StreamHandler()
# CH.setLevel(logging.DEBUG)
# CH.setFormatter(FORMATTER)
# LOG.addHandler(CH)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(name)-8s - %(levelname)-8s: %(message)s')
LOG = logging.getLogger(__name__)


def from_cfg(cfg_fnm):
    '''Start all language model servers'''

    with open(cfg_fnm, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    active_langs = [l.strip() for l in cfg['serve_langs'].split(',')]
    langs = [cfg['servers'][lang] for lang in active_langs]

    # Start each language server in separate process. These will stay alive indefinitely
    server_procs = []
    for lang in langs:
        proc = os.fork()
        if proc == 0:
            # I'm a child process: start a server
            LOG.info("Starting server for lang {}".format(lang['spacy_lang']))
            server.serve(host=lang['host'], port=lang['port'])
        else:
            # I'm the parent: keep track of child processes
            server_procs.append(proc)

    # Use client to specify model and use a request to prevent lazy-loading later on
    # For speed up, also do this in parallel processes. These die immediately after initiation of models.
    time.sleep(2)
    load_procs = []
    for lang in langs:
        host = lang['host']
        port = lang['port']
        spacy_lang = lang['spacy_lang']
        proc = os.fork()
        if proc == 0:
            try:
                LOG.info("Initiating client for lang {}".format(lang['spacy_lang']))
                if lang['wv_pretrained_fnm'] is not None:
                    api_client = client.Client(host=host, port=port, model=spacy_lang,
                                               embeddings_path=lang['wv_pretrained_fnm'], verbose=True)
                else:
                    api_client = client.Client(host=host, port=port, model=spacy_lang, verbose=True)
                api_client.single('dummy')
                LOG.info("{} server ready on {}:{}".format(spacy_lang, host, port))
            except Exception as e:
                LOG.error("Encountered exception. Cannot initialize server...")
                LOG.exception(e)
            finally:
                os._exit(1)
        else:
            load_procs.append(proc)

    # Wait until all models are initiated
    for proc in load_procs:
        os.waitpid(proc, 0)

    LOG.info("All language servers ready, running in these processes: {}".format(server_procs))
    LOG.info("Parent process id: {}".format(os.getpid()))
    LOG.info("Kill me to shut down all language servers...")

    # Keep parent alive as long as any of the servers are alive. Needs to be explicitly killed.
    for proc in server_procs:
        os.waitpid(proc, 0)
