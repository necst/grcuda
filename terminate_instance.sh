# the name of the instance to terminate is passed as the first argument of the script
display_name=$1

# example OCID for grcuda compartment, substitute with OCID of your compartment
compartment_id=ocid1.compartment.oc1..aaaaaaaakrrqfga2bqdbv7ruydhh36cbyy46w2m6k4dnbdfw3gyes45qebfq 

# get instance id
instance_id=$(oci compute instance list -c $compartment_id --lifecycle-state RUNNING --display-name $display_name --query data[0].id --raw-output)

# print info (name, id) about the instance to terminate
echo display-name=$1
echo id=$instance_id

# terminate instance (the terminate command automatically ask for confirmation)
# set --preserve-boot-volume to false if you want to permanently erase the boot volume attached to the instance
oci compute instance terminate --instance-id $instance_id --preserve-boot-volume true --wait-for-state TERMINATED