# This is the source file for queue modelling of a drone swarm processing incoming stream of data packets.
# This implementation includes individual queuing, packet priorities and different processing times

import numpy as np
import random
from matplotlib import pyplot as plt
from typing import List, Tuple, Dict, Callable
import math


class DataPacket:
    def __init__(self, priority: int=1, lifetime: int=100, steps_to_complete: int=1, i: int=None):
        """Initialize a data packet with set priority, lifetime duration and number of steps to complete

        Args:
            priority (int): Priority level; bigger value means bigger priority and quicker queuing
            lifetime (int): Lifetime duration of the packet; after this number of steps the packet is useless
            steps_to_complete (int): Number of processing steps needed for full processing of the packet
            i (int): unique identifier of this packet
        """
        self.priority = priority
        self.lifetime = lifetime
        self.steps_to_complete = steps_to_complete
        self.initial_lifetime = self.lifetime
        self.initial_steps_to_complete = self.steps_to_complete
        self.i = i
        assert self.lifetime > 0 and self.steps_to_complete > 0

    def iteration_wait(self) -> None:
        """This happens to the packet each iteration
        """
        self.lifetime -= 1

    def is_obsolete(self) -> bool:
        """Check if the packet is no longer useful
        """
        return self.lifetime <= 0

    def __repr__(self):
        """For printing
        """
        return f'P<{self.i}>({self.priority}, {self.lifetime}/{self.initial_lifetime}, {self.steps_to_complete}/{self.initial_steps_to_complete})'


def packet_list_steps(packets: List[DataPacket]) -> int:
    """Sum of steps-to-complete of all packets in the list
    """
    return sum([p.steps_to_complete for p in packets])
    

class Drone:
    def __init__(self, queue_size: int=1000, power: int=10):
        """Initialize a drone with queue size and power

        Args:
            queue_size (int): how many packets the drone can hold in a queue
            power (int): power, measured in processing steps per iteration
        """
        self.queue_size = queue_size
        self.queue: List[DataPacket] = []
        self.power = power

    def assign_packets(self, packets: List[DataPacket]) -> List[DataPacket]:
        """Add packets to drone's queue, handling different priorities.
        Returns packets that haven't been assigned
        """
        temp_packets = packets[:]
        while len(self.queue) < self.queue_size and len(temp_packets):
            packet = temp_packets.pop(0)
            self._queue_packet(packet)
        return temp_packets

    def _queue_packet(self, packet: DataPacket) -> int:
        """Insert a single packet into the queue, taking its priority into account.
        Returns new queue size
        """
        if not len(self.queue):
            # If the queue is empty - just insert
            self.queue.append(packet)
        else: 
            if packet.priority > max([p.priority for p in self.queue]):
                # If the packet has higher priority than all packets 
                # in the queue - insert last (first to process)
                self.queue.append(packet)
            elif packet.priority <= min([p.priority for p in self.queue]):
                # If the packet has lower priority than all packets
                # in the queue - insert first (last to process)
                self.queue.insert(0, packet)
            else:
                # Otherwise find where to insert the packet
                for i in range(len(self.queue)):
                    if packet.priority <= self.queue[i].priority:
                        self.queue.insert(i, packet)
                        break
        return len(self.queue)

    def is_idle(self) -> bool:
        """Check if drone's queue is empty
        """
        return len(self.queue) == 0

    def iteration(self) -> Tuple[List[DataPacket], List[DataPacket]]:
        """The action a drone executes on each iteration: processing of packets withing the queue.
        Returns the list of processed packets and the list of packets that became obsolete
        """
        # Remove packets that are obsolete
        obsolete_packets = [packet for packet in self.queue if packet.is_obsolete()]
        self.queue = [packet for packet in self.queue if not packet.is_obsolete()]
        
        # If queue is empty - nothing to do this iteration, zero packets have been processed
        if not len(self.queue):
            return [], obsolete_packets
        
        current_power = self.power

        # Process the packets in the queue
        processed_packets = []
        for i in range(len(self.queue)-1, -1, -1):
            if current_power >= self.queue[i].steps_to_complete:
                # [i]th packet can be processed fully
                current_power -= self.queue[i].steps_to_complete
                self.queue[i].steps_to_complete = 0
                processed_packets.append(self.queue[i])
                # If this was the last packet in the queue, and drone still has power,
                # immediately empty the queue
                if i == 0:
                    self.queue = []
                    break
            else: 
                if current_power > 0:
                    # [i]th packet can be processed partially
                    self.queue[i].steps_to_complete -= current_power
                    current_power = 0
                    # Remove fully processed packets from the queue, keeping the [i]th
                    self.queue = self.queue[:i+1]
                    break
                else:
                    # Nothing can be done anymore, drone power is exhausted
                    # Remove fully processed packets from the queue, keeping the [i]th
                    self.queue = self.queue[:i+1]
                    break

        # Pass the time
        for i in range(len(self.queue)):
            self.queue[i].iteration_wait()

        queue_ids = [p.i for p in self.queue]
        processed_ids = [p.i for p in processed_packets]
        if set(queue_ids).intersection(set(processed_ids)) != set():
            print('')

        # Return processed packets
        return processed_packets, obsolete_packets

    def steps_in_queue(self) -> int:
        """Sum of steps-to-complete of all packets currently in queue
        """
        return sum([p.steps_to_complete for p in self.queue])
    
    def __repr__(self):
        """For printing
        """
        return f'Drone (queue({len(self.queue)})={self.queue}, power={self.power})'


class StreamGenerator:
    def __init__(self, priorities: List[int], probs: List[float]=None, 
                 steps_from_priority: Callable[[int], int]=lambda p: p, 
                 lifetime_from_priority: Callable[[int], int]=lambda p: max(3, 5-p),
                 batch_size: int=100, batch_var: int=0):
        """Initialize a packet stream generator with list of possible priorities, their probabilities,
        rule by which a packet's complexity (steps to complete) and lifetime depend on its priority and batch size

        Args:
            priotities (List[int]): list of possible priority levels of packets
            probs (List[float]): list of probabilities of priorities that must sum up to 1.0 or None, if all equally possible
            steps_from_priority (Callable[[int], int]): rule that defines a packet's steps complexity S from its priority P as S = steps_from_priority(P)
            lifetime_from_priority (Callable[[int], int]): rule that defines a packet's lifetime, same as with steps_from_priority
            batch_size (int): the mean size of generated batches
            batch_var (int): size of actual generated batches will fluctuate in (batch_size - batch_var, batch_size + batch_var)
        """
        self.priorities = priorities
        self.probs = probs
        self.steps_from_priority = steps_from_priority
        self.lifetime_from_priority = lifetime_from_priority
        self.batch_size = batch_size
        self.batch_var = batch_var

        self.last_packet_i = 0

        if self.probs is not None:
            if len(self.probs) != len(self.priorities):
                raise Exception('List of probabilities must be of same size as list of priorities')
            if sum(self.probs) != 1.0:
                raise Exception(f'Sum of probabilities must be equal to 1.0, not {sum(self.probs)}')
    
    def generate_batch(self) -> List[DataPacket]:
        """Generate the stream as a list of packets and return it
        """
        stream = []
        this_batch_size = self.batch_size + int(random.uniform(-self.batch_var, self.batch_var+0.1)) 
        priorities = random.choices(self.priorities, weights=self.probs, k=this_batch_size)
        for p in priorities:
            # Calculate packet's steps-to_complete from its priority
            steps_to_complete = self.steps_from_priority(p)
            # Calculate packet's lifetime from its priority
            lifetime = self.lifetime_from_priority(p)
            # Create packet and add to the stream list
            stream.append(DataPacket(
                priority=p,
                lifetime=lifetime,
                steps_to_complete=steps_to_complete,
                i=self.last_packet_i
            ))
            self.last_packet_i += 1
        return stream


class SwarmManager:
    def __init__(self, drone_powers: List[int], drone_queue_size: int, stream_generator: StreamGenerator):
        """Initialize the swarm manager, that consists of:
        1. Swarm itself: number of drones
        2. Packet stream

        Args:
            drone_powers (List[int]): list of drones' powers, measured in processing steps per iteration; indirectly defines drones count
            drone_queue_size (int): how many packets drones can keep queued; same for all drones
            stream_generator (StreamGenerator): this component generates a sequence of packets from given parameters
        """
        self.drones: List[Drone] = []
        for dp in drone_powers:
            self.drones.append(Drone(
                queue_size=drone_queue_size,
                power=dp
            ))
        self.stream_generator = stream_generator

    def distribute_packets(self, packets: List[DataPacket]) -> List[DataPacket]:
        """Distributes packets from a list among the drones; returns remaining packets that haven't been assigned 
        """
        # First, assign packets to drones with empty queues
        temp_packets = packets[:]
        for i in range(len(self.drones)):
            if self.drones[i].is_idle():
                # Assign packets
                if not len(temp_packets):
                    break
                temp_packets = self.drones[i].assign_packets(temp_packets)
        
        # Now, assign packets to drones with more processing power first
        power_indices = [i for i, _ in sorted(enumerate(self.drones), key=lambda x: x[1].power, reverse=True)]
        for i in power_indices:
            if not len(temp_packets):
                break
            temp_packets = self.drones[i].assign_packets(temp_packets)
        return temp_packets

    def do_iteration(self) -> Tuple[Dict[int, List], Dict[int, List]]:
        """Make each drone do one iteration of processing; returns dict of drone indices -> processed packets
        """
        processed_packets = {}
        obsolete_packets = {}
        for i in range(len(self.drones)):
            processed_packets[i], obsolete_packets[i] = self.drones[i].iteration()
        return processed_packets, obsolete_packets

    def describe_drones_state(self) -> str:
        """Create a string description of the drone's current state
        """
        description = []
        for i, drone in enumerate(self.drones):
            drone_str = f'Drone #{i}: '
            if drone.is_idle():
                drone_str += 'idle'
            else:
                p = drone.queue[-1]
                drone_str += f'on {str(p)}'
            description.append(drone_str)
        return ' | '.join(description)

    def average_queue_load(self) -> float:
        """Average queue load of all drones as a fraction
        """
        total_queues_capacity = sum([d.queue_size for d in self.drones])
        total_queues_load = sum([len(d.queue) for d in self.drones])
        return total_queues_load / total_queues_capacity

    def total_power(self) -> int:
        """Sum of powers of all drones, measured in processing steps per iteration
        """
        return sum([d.power for d in self.drones])

    def total_queues_size(self) -> int:
        """Sum of all drones' queues capacities
        """
        return sum([d.queue_size for d in self.drones])

    def run_simulation(self, n_iterations: int, verbose: bool=True) -> List[Dict]:
        """Run the simulation for given number of iterations. Each iteration a new batch of packets arrives and is distributed among 
        drones, they do their processing and produce processed packets. Returns list of dicts with information of the simulation

        Args:
            n_iterations (int): the simulation will run for this many iterations
            verbose (bool): if True, print each iteration's statistics
        """
        total_power = self.total_power()
        total_queues_size = self.total_queues_size()
        print(f'Swarm total power: {total_power} processing steps per iteration')
        print(f'Swarm total queues capacity: {total_queues_size} packets')
        sim_info = []
        for i in range(n_iterations):
            iteration_info = {}
            # 1. Generate a batch of packets for this iteration
            packets = self.stream_generator.generate_batch()
            packet_count = len(packets)
            total_steps = packet_list_steps(packets)
            # 2. Assign packets to drones
            remaining_packets = self.distribute_packets(packets)
            assigned_count = len(packets) - len(remaining_packets)
            remaining_count = len(remaining_packets)
            queues_load = self.average_queue_load()
            # 3. Do one iteration of processing
            processed_packets, obsolete_packets = self.do_iteration()            
            processed_count = sum([len(p) for d_i, p in processed_packets.items()])
            obsolete_count = sum([len(p) for d_i, p in obsolete_packets.items()])
            # 4. Information of this iteration
            iteration_info['packet_count'] = packet_count
            iteration_info['total_steps'] = total_steps
            iteration_info['assigned_count'] = assigned_count
            iteration_info['remaining_count'] = remaining_count
            iteration_info['processed_count'] = processed_count
            iteration_info['obsolete_count'] = obsolete_count
            iteration_info['queues_load'] = queues_load
            # 5. Display current state
            if verbose:
                print(f'{i+1:>3}. Generated: {packet_count:>3} ({total_steps:>3} steps), assigned: {assigned_count:>3}, remaining: {remaining_count:>3}, '
                    f'processed: {processed_count:>3}, obsolete: {obsolete_count:>3}, average queues capacity: {100*queues_load:.2f}%')
            sim_info.append(iteration_info)
        return sim_info


def print_general_info(sim_info: List[Dict]):
    """Print information of the simulation overall
    """
    total_packets = sum([iter_info['packet_count'] for iter_info in sim_info])
    processed_count = sum([iter_info['processed_count'] for iter_info in sim_info])
    processed_percent = 100.0 * processed_count / total_packets
    remaining_count = sum([iter_info['remaining_count'] for iter_info in sim_info])
    remaining_percent = 100.0 * remaining_count / total_packets
    obsolete_count = sum([iter_info['obsolete_count'] for iter_info in sim_info])
    obsolete_percent = 100.0 * obsolete_count / total_packets
    print(f'\nDuring {len(sim_info)} iterations:')
    print(f'\t{total_packets} packets generated,')
    print(f'\t{processed_count} processed ({processed_percent:.2f}%),')
    print(f'\t{remaining_count} lost due to overload ({remaining_percent:.2f}%),')
    print(f'\t{obsolete_count} lost due to timeouts ({obsolete_percent:.2f}%)')
    print(f'\tThe rest are currently in drones\' queues')


def main():
    # 1. Create the packet stream generator
    # Buradaki fonksiyonları değiştirerek farklı senaryoları test edeceğiz

# Kural 1:
#     stream_generator = StreamGenerator(
#     priorities=[1, 2, 3],
#     probs=[0.4, 0.3, 0.3],
#     steps_from_priority=lambda p: max(1, p+2),
#     lifetime_from_priority=lambda p: max(1, 6-p),
#     batch_size=100,
#     batch_var=5
# )

# #Kural 2:
#     stream_generator = StreamGenerator(
#     priorities=[1, 2, 3],
#     probs=[0.4, 0.3, 0.3],
#     steps_from_priority=lambda p: p**2,
#     lifetime_from_priority=lambda p: 10-p,
#     batch_size=100,
#     batch_var=5
# )

# Kural 3:
#     stream_generator = StreamGenerator(
#     priorities=[1, 2, 3],
#     probs=[0.4, 0.3, 0.3],
#     steps_from_priority=lambda p: 2*p,
#     lifetime_from_priority=lambda p: 5 + p,
#     batch_size=100,
#     batch_var=5
# )

# Kural 4:
    stream_generator = StreamGenerator(
    priorities=[1, 2, 3],
    probs=[0.4, 0.3, 0.3],
    steps_from_priority=lambda p: 3 if p > 1 else 1,
    lifetime_from_priority=lambda p: max(2, 8-p),
    batch_size=100,
    batch_var=5
    
)


    # 2. Create the swarm manager
    drone_queue_size = 20
    swarm_manager = SwarmManager(
        drone_powers=[10, 15, 20, 12, 8, 12, 15, 20],
        drone_queue_size=drone_queue_size,
        stream_generator=stream_generator
    )
    # 3. Run the simulation
    n_iterations = 100
    sim_info = swarm_manager.run_simulation(n_iterations, verbose=False)


    # 4. Information of the simulation overall
    print_general_info(sim_info)
    # 5. Visualization
    packets_by_iteration = [iter_info['packet_count'] for iter_info in sim_info]
    queues_by_iteration = [iter_info['queues_load'] * drone_queue_size for iter_info in sim_info]
    processed_by_iteration = [iter_info['processed_count'] for iter_info in sim_info]
    plt.figure(figsize=(12, 8))
    plt.plot(range(n_iterations), packets_by_iteration, label='Received data packets')
    plt.plot(range(n_iterations), processed_by_iteration, label='Processed data packets')
    plt.plot(range(n_iterations), [drone_queue_size] * n_iterations, label='Queues maximum fullness')
    plt.plot(range(n_iterations), queues_by_iteration, label='Queues average fullness')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
            

        

