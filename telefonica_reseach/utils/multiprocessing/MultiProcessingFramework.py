# https://pymotw.com/3/multiprocessing/communication.html
# This script is produced to see if the average of all points in the latent space is zero

import multiprocessing as mp
import json
from glob import glob
import time


class Processor(mp.Process):
    def __init__(self, name=None, total_num_task=None):
        mp.Process.__init__(self)
        self.task_queue: mp.JoinableQueue = None
        self.result_queue: mp.Queue = None
        if name:
            self.name = name
        self.total_num_task = total_num_task

    def run(self):
        proc_name = self.name

        print('starting:{}'.format(proc_name))

        while True:
            # {'img': imgs, 'num_task': num_tasks, 'total_num_task': gp_file_total}
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                global_result = self.process_global_task()
                if global_result:
                    self.result_queue.put(global_result)

                self.task_queue.task_done()
                break

            print('{}: {} / {}'.format(proc_name, next_task['id'], self.total_num_task))

            result = self.process_task(next_task['task'])

            if result:
                self.result_queue.put(result)

            self.task_queue.task_done()
        return

    def process_task(self, wrapper_task):
        pass

    def process_global_task(self):
        pass


class MultiProcessManager(object):

    def __init__(self):
        self.processors_list = []
        self.tasks = mp.JoinableQueue()
        self.results = mp.Queue()
        self.tasks_id = 0
        self.hash_broadcasted = False

    def add_processor(self, new_processor_):
        new_processor = new_processor_()
        new_processor.task_queue = self.tasks
        new_processor.result_queue = self.results
        if not new_processor.name:
            new_processor.name = str(len(self.processors_list)+1)
        self.processors_list.append(new_processor)
        new_processor.start()

    def send_task(self, task):
        task_wrapper = {}
        self.tasks_id += 1
        task_wrapper['id'] = self.tasks_id
        task_wrapper['task'] = task
        self.tasks.put(task_wrapper)

    def broadcast_end(self):
        for i in range(len(self.processors_list)):
            self.tasks.put(None)

    def wait(self):
        self.tasks.join()

    def next_result(self):
        if not self.hash_broadcasted:
            self.broadcast_end()
            self.hash_broadcasted = True
            self.wait()
            time.sleep(1)

        if not self.results.empty():
            return self.results.get()
        else:
            return None

    def get_max_processors_num(self):
        return mp.cpu_count() * 2 - 2


def chunks(yield_lst, chunk_size=100):
    """Yield successive n-sized chunks from lst."""
    chunked_list = []
    list_size = 0

    for e in yield_lst:
        chunked_list.append(e)
        list_size += 1

        if list_size > chunk_size:
            chunked_list = [e]
            list_size = 1

        if list_size == chunk_size:
            yield chunked_list

    yield chunked_list




class MyProcessor(Processor):

    def __init__(self):
        Processor.__init__(self)
        self.my_count = 0
        self.my_value = 0

    def process_task(self, task):
        self.my_count += 1
        self.my_value += task['value']
        return None

    def process_global_task(self):
        result = {}
        result['value'] = self.my_value
        result['count'] = self.my_count

        return result


if __name__ == '__main__':

    mpm = MultiProcessManager()

    for r in range(0, 1):
        mpm.add_processor(MyProcessor)

    for r in range(0, 2):
        mpm.send_task({'value': r})

    result_global = 0
    count_global = 0

    while True:
        result = mpm.next_result()

        if result:
            result_global += result['count']*result['value']
            count_global += result['count']
        else:
            print(result_global/count_global)
            print('hola')
            break
