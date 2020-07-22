#!env python
# -*- coding: utf-8 -*-

import argparse
import base64
import collections
import logging
import math
import multiprocessing
import os
import queue
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from multiprocessing import Queue
from multiprocessing.managers import SyncManager

import numpy as np


class slurmmanager:
    """This class handles the distributed calculation of a scoring function using SLURM"""

    def __init__(self, nb_local=8, nb_slurm=0, cpu_per_job=8, scoring_function="rocs_similarity", **kwargs):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            self._port = s.getsockname()[1]
        self._authkey = base64.b64encode(os.urandom(32)).decode()
        self.args = kwargs
        self.args["scoring_function"] = scoring_function
        self.nb_local = 0
        self.nb_slurm = 0
        self._local_worker = []
        self.cpu_per_job = cpu_per_job
        self.smiles_per_worker = 1 * cpu_per_job
        self.slurmids = []
        self.jobnonce = 0

        def make_server_manager(port, authkey, args):
            """ Create a manager for the server, listening on the given port.
                Return a manager object with get_job_q and get_result_q methods.
            """

            # This is based on the examples in the official docs of multiprocessing.
            # get_{job|result}_q return synchronized proxies for the actual Queue
            # objects.
            class JobQueueManager(SyncManager):
                pass

            job_q = multiprocessing.Queue()
            result_q = multiprocessing.Queue()

            JobQueueManager.register('get_job_q', callable=lambda: job_q)
            JobQueueManager.register('get_result_q', callable=lambda: result_q)
            JobQueueManager.register('arguments', callable=lambda: args)
            manager = JobQueueManager(address=('', port), authkey=authkey.encode())
            manager.start()
            logging.debug('Server started at port %s' % port)
            return manager

        self._manager = make_server_manager(port=self._port, authkey=self._authkey, args=self.args)
        self._job_q = self._manager.get_job_q()
        self._result_q = self._manager.get_result_q()
        self._tasks = collections.OrderedDict()
        self.addSlurmWorker(int(math.ceil(nb_slurm / cpu_per_job)))
        self.addLocalWorker(nb_local)
        time.sleep(10)  # give slum and the master some time to start the worker

    def addTasks(self, tasklist):
        """ Adds the elements of the tasklist into the job queue."""
        assert isinstance(tasklist, list)
        for i in range(0, len(tasklist), self.smiles_per_worker):
            elements = tasklist[i:i + self.smiles_per_worker]
            for element in elements:
                self._tasks[element] = None
            self._job_q.put((elements, self.jobnonce))

    def getResults(self, timeout=0):
        """ Returns a dictionary with results """

        if timeout <= 0:
            timeout = float("inf")
        else:
            # We extend the timeput in case not all slurm worker are running
            if self.runningSlurmWorker() < self.nb_slurm:
                logging.debug("Not all SLURM worker are running. Disabling the calculation timeout.")
                timeout = float("inf")
        start_t = time.time()
        while (None in self._tasks.values()) and (time.time() - start_t < timeout):
            try:
                results = self._result_q.get(timeout=1)
                if isinstance(results, tuple):
                    results, jobnonce = results
                    if isinstance(results, dict) and self.jobnonce == jobnonce:
                        for key, value in results.items():
                            if key in self._tasks:
                                self._tasks[key] = value
            except queue.Empty:
                pass

        results = {key: value for key, value in self._tasks.items()}
        self._tasks = collections.OrderedDict()
        self.jobnonce += 1
        return results

    def __call__(self, smiles) -> np.ndarray:
        if self._job_q.qsize() > 0:
            self._emptyjob_q()
        self.addTasks(smiles)
        results = self.getResults()
        scores = [results[smi] if results[smi] is not None else 0.0 for smi in smiles]
        return np.array(scores, dtype=np.float32)

    def _emptyjob_q(self):
        try:
            while not self._job_q.empty():
                try:
                    _ = self._job_q.get_nowait()
                except queue.Empty:
                    break
        except:  # connection errors etc.
            pass

    def __del__(self):
        """
        Remove all scheduled jobs and spam the TaskQueue with NONE such the worker can stop gracefully.
        Calls also scancel to kill the jobs otherwise
        """
        self._emptyjob_q()
        for _ in range((self.nb_local + self.nb_slurm) * self.smiles_per_worker * 2):
            try:
                self._job_q.put_nowait(None)
            except:
                pass
        time.sleep(5)
        for jobid in self.slurmids:
            logging.debug("Run: " + " ".join(["scancel", jobid]))
            output = subprocess.check_output(["scancel", jobid])
            logging.debug(output)
        self._manager.shutdown()

    def runningSlurmWorker(self):
        output = subprocess.check_output(["sacct", "--jobs=" + ','.join(self.slurmids), "--format=State", "--noheader"])
        lines = output.split()
        running = 0
        for line in lines:
            if line.decode() == "RUNNING":
                running += 1
        return running

    def addLocalWorker(self, nb_worker):
        """ Starts new local processes as worker """
        logging.debug("Start local worker")
        p = multiprocessing.Process(target=runworker, args=("localhost", self._port, self._authkey, nb_worker))
        p.start()
        self._local_worker.append(p)
        self.nb_local += 1

    def addSlurmWorker(self, nb_jobs):
        """ Creates an sbatch script and submit it to slurm. When the jobs are scheduled they connect to
         the master process """
        cmd = "PYTHONUNBUFFERED=1 " + sys.executable + " " + __file__ + " --hostname " + socket.gethostname() \
              + " --port " + str(self._port) + " --authkey " + self._authkey + " --cpu " + str(self.cpu_per_job)
        cmd = shlex.split(cmd)
        cmdline = " ".join(cmd)

        slurm_script = ["#!/bin/bash",
                        "#SBATCH --job-name=reinvent-worker",
                        "#SBATCH --mem-per-cpu=400",
                        "#SBATCH --nodes=1",
                        "#SBATCH --ntasks=1",
                        "#SBATCH --hint=nomultithread",
                        "#SBATCH --cpus-per-task {}".format(self.cpu_per_job),
                        "#SBATCH --time=0-12:00:00",
                        cmdline]
        tempfolder = tempfile.mkdtemp()
        logging.debug("Write slurm script with following content: {}".format("\n".join(slurm_script)))
        with open(os.path.join(tempfolder, "reinvent_worker.sh"), 'w') as f:
            for line in slurm_script:
                print(line, file=f)

        loggerfolder = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
        for i in range(nb_jobs):
            try:
                logging.debug("Run: " + " ".join(["sbatch", "--output", os.path.join(loggerfolder, "slurm-%j.out"),
                                                  os.path.join(tempfolder, "reinvent_worker.sh")]))
                output = subprocess.check_output(["sbatch", "--output", os.path.join(loggerfolder, "slurm-%j.out"),
                                                  os.path.join(tempfolder, "reinvent_worker.sh")])
                output = output.decode()
                slurmid = output.strip().split(" ")[-1]
                logging.debug(output)
                self.nb_slurm += 1
                self.slurmids.append(slurmid)
            except subprocess.CalledProcessError as e:
                logging.info(e)
        os.remove(os.path.join(tempfolder, "reinvent_worker.sh"))
        os.rmdir(tempfolder)


def runworker(hostname, port, authkey, cores):
    class ServerQueueManager(SyncManager):
        pass

    ServerQueueManager.register('get_job_q')
    ServerQueueManager.register('get_result_q')
    ServerQueueManager.register('arguments')

    maxtrys = 5
    while maxtrys >= 0:
        try:
            manager = ServerQueueManager(address=(hostname, port), authkey=authkey.encode())
            manager.connect()
            print("Sucessfully connected")
            break
        except:
            # probably our server is not up yet
            print("Couldn't connect. Retry.")
            maxtrys -= 1
            time.sleep(5)

    job_q = manager.get_job_q()

    result_q = manager.get_result_q()

    args = manager.arguments()
    args = {key: args.get(key) for key in args.keys()}
    import scoring
    import multiprocessing
    scoring_function = args.pop("scoring_function")
    scoring_function = scoring.get_scoring_function(scoring_function, **args)
    pool = multiprocessing.Pool(cores)
    
    maxtry = 10
    while True:
        try:
            jobmessage = job_q.get(block=True)
            maxtry = 10
            if jobmessage is None:
                return 0
            elif isinstance(jobmessage, tuple):
                job, jobnonce = jobmessage
                # job is a list of smiles
                if len(job) < cores:  # we can try to get more jobs
                    enough = False
                    while not enough:
                        try:
                            additional_job = job_q.get_nowait()
                            if additional_job is None:
                                return 0
                            job.extend(additional_job)
                            if len(job) < cores:
                                enough = True
                        except queue.Empty:
                            # There is nothing left, we can just start to process
                            enough = True
                smiles = [[smi] for smi in job]
                try:
                    scores_arr = pool.map(scoring_function, smiles, chunksize=1)
                except:
                    print("Error in omega")
                scores = [s[0] for s in scores_arr]
                outdict = dict(zip(job, scores))
                result_q.put((outdict, jobnonce))
        except Exception as e:
            print("Socket error. Message: {}".format(e))
            # probably our connection died
            maxtry -= 1
            if maxtry <= 0:
                return 1


def get_reinvent_path():
    import os
    scoring = os.path.dirname(os.path.realpath(__file__))
    return "/".join(scoring.split("/")[0:-1])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", "-s", help="Hostname to connect to", type=str)
    parser.add_argument("--port", "-p", help="Port for connecting. Default: 31992", type=int, default=31992)
    parser.add_argument("--authkey", "-k", help="Authkey for connecting", type=str)
    parser.add_argument("--cpu", "-c", help="Number of cores to use. Default: 8", type=int, default=8)
    args = parser.parse_args()
    import sys
    sys.path.append(get_reinvent_path())
    print(sys.path)
    print("Start worker with \n --hostname {} \n --port {} \n --authkey {} \n --cpu {}".format(args.hostname, args.port,
                                                                                               args.authkey, args.cpu))
    exitcode = runworker(args.hostname, args.port, args.authkey, args.cpu)
    if exitcode == 0:
        print("Received stop signal from master.")
    else:
        print("Connection died.")
