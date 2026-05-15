from review_simulator import ReviewSimulator

sim = ReviewSimulator(
      profiles_path="./data/user_profiles.json",
      items_path="./data/item_profiles.json"
  )
result = sim.simulate(
      user_id="-V7frksbFjKQYVhrPnnlPw",
      business_id="oQ5CPRt0R3AzFvcjNOqB1w",
      context={}
  )
# evaluation against ground truth
# results = sim.evaluate(
#       held_out_path="./data/held_out_pairs.json",
#       output_path="./data/evaluation_results.json")
print (result)