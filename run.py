from PPLM.PPLM import run_pplm

run_pplm(
        pretrained_model="gpt2",
        cond_text="buy the galaxy s21 Ultra phone",
        discrim='sentiment',
        uncond=False,
        verbosity = 'very_verbose'
)