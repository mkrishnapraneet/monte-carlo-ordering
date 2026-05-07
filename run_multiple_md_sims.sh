for i in {1..105}
do
  python3 short-traj.py 80 events_inputs/multiple-short-trajectory_${i} &
done