from operator import add
from typing import Optional, Tuple
from matplotlib.pyplot import axis

import numpy as np
import torch, os
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from PPLM.Discriminator import ClassificationHead
from utils.params import bcolors, params, PPLM as paramspplm

from data import getResonanceInfo

PPLM_DISCRIM = 2
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}


def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def get_classifier(
        name: Optional[str],
        class_label: str,
        device: str
) -> Tuple[Optional[ClassificationHead], Optional[int]]:

    if name is None:
        return None, None

    model_params = paramspplm.DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=model_params['class_size'],
        embed_size=model_params['embed_size']
    ).to(device)
    
    if "path" in model_params:
        resolved_archive_file = model_params["path"]
    else:
        raise ValueError(f"{bcolors.FAIL}{bcolors.BOLD}Enter the pretrained discriminator path!{bcolors.ENDC}")
    classifier.load(resolved_archive_file, device)
    classifier.eval()

    return classifier, (1 if class_label=='pos' else 0)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_past(
        past,
        model,
        last,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        classifier=None,
        class_label=None, 
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR,
        content_guide=None,
        semantic_weight = 0.2
):
    # Generate inital perturbed past

    grad_accumulator = [
        np.zeros(p.shape).astype("float32")
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
                 tuple(past[0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape


        model_output = model(last, past_key_values=perturbed_past) 
        all_logits, all_hidden = model_output.logits, model_output.hidden_states

        hidden = all_hidden[-1]

        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []

        ce_loss = torch.nn.CrossEntropyLoss()
        
        curr_unpert_past = unpert_past
        curr_probs = torch.unsqueeze(probs, dim=1)
        wte = model.resize_token_embeddings()
        for _ in range(horizon_length):
            inputs_embeds = torch.matmul(curr_probs, wte.weight.data)

            current_model_output = model( past_key_values=curr_unpert_past, inputs_embeds=inputs_embeds)

            curr_unpert_past, curr_all_hidden = current_model_output.past_key_values, current_model_output.hidden_states
            curr_unpert_past = [ torch.cat([_p.unsqueeze(0) for _p in layer]) for layer in curr_unpert_past]

    
            curr_hidden = curr_all_hidden[-1]

            new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                curr_hidden, dim=1)

        prediction = classifier(new_accumulated_hidden /
                                (curr_length + 1 + horizon_length))

        label = torch.tensor(prediction.shape[0] * [class_label],
                              device=device,
                              dtype=torch.long)

        if content_guide is not None:
          discrim_loss = (1.0 - semantic_weight)*ce_loss(prediction, label) - \
                      semantic_weight*torch.cosine_similarity(content_guide, new_accumulated_hidden/(curr_length + 1 + horizon_length))
                      #semantic_weight*torch.sqrt(torch.sum((content_guide - new_accumulated_hidden/(curr_length + 1 + horizon_length))**2))
        else:
          discrim_loss = ce_loss(prediction, label)

        if verbosity_level >= VERY_VERBOSE:
            print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
        loss += discrim_loss
        loss_list.append(discrim_loss)

        #! Removed loss type discrimination

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                print(' kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        grad_norms = [
            (torch.norm(p_.grad * window_mask) + SMALL_CONST)
            for index, p_ in enumerate(curr_perturbation)
        ]

        # normalize gradients
        grad = [
            -stepsize *
            (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def generate_text_pplm(
        model,
        tokenizer,
        context=None,
        past=None,
        device="cuda",
        perturb=True, 
        classifier=None,
        class_label=None, 
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        content_guide=None,
        semantic_weight=None
):
    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    if verbosity_level >= VERBOSE:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)

    for i in range_func:

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                past = model(output_so_far[:, :-1]).past_key_values
                past = [ torch.cat([_p.unsqueeze(0) for _p in layer]) for layer in past]

        model_output = model(output_so_far)
        unpert_logits, unpert_past, unpert_all_hidden = model_output.logits, model_output.past_key_values, model_output.hidden_states
        unpert_past = [ torch.cat([_p.unsqueeze(0) for _p in layer]) for layer in unpert_past]
        
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:

                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize, 
                    classifier=classifier,
                    class_label=class_label, 
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                    verbosity_level=verbosity_level,
                    content_guide=content_guide,
                    semantic_weight=semantic_weight
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        model_output = model(last, past_key_values=pert_past)
        pert_logits, past, pert_all_hidden = model_output.logits, model_output.past_key_values, model_output.hidden_states
        past = [ torch.cat([_p.unsqueeze(0) for _p in layer]) for layer in past]

        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([class_label], device=device,
                                 dtype=torch.long)
            unpert_discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERBOSE:
                print(
                    "unperturbed discrim loss",
                    unpert_discrim_loss.data.cpu().numpy()
                )
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k,
                                      probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )
        if verbosity_level >= VERBOSE:
            print(tokenizer.decode(output_so_far.tolist()[0]))
    return output_so_far, unpert_discrim_loss, loss_in_time

def full_text_generation(
        model,
        tokenizer,
        context=None,
        num_samples=1,
        device="cuda",
        discrim=None,
        class_label=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        semantic_weight=0.2,
        **kwargs
):
  classifier, class_id = get_classifier(
    discrim,
    class_label,
    device
    )

  if classifier is not None:
    print(f"{bcolors.OKBLUE}{bcolors.BOLD}Using PPLM-Discrim {discrim.upper()} {bcolors.ENDC}")
  else: 
    print(f"{bcolors.FAIL}{bcolors.BOLD}Specify a Discriminator{bcolors.ENDC}")
    exit(1)

  unpert_gen_tok_text, _, _ = generate_text_pplm( #! This is the unperturbed manner
      model=model,
      tokenizer=tokenizer,
      context=context,
      device=device,
      length=length,
      sample=sample,
      perturb=False,
      verbosity_level=verbosity_level
  )
  
  hidden_sates_unperturbed = model(unpert_gen_tok_text).hidden_states
  latent_generation_unpert = torch.mean(hidden_sates_unperturbed[-1].detach(), axis=1)


  if device == 'cuda':
    torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

  latent_generation_pert_sim = []
  for i in range(num_samples):
    pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm( 
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        perturb=True,
        classifier=classifier,
        class_label=class_id,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity_level=verbosity_level,
        content_guide=latent_generation_unpert,
        semantic_weight=semantic_weight
    )

    pert_gen_tok_texts.append(pert_gen_tok_text)
    hidden_sates_perturbed = model(pert_gen_tok_text).hidden_states
    latent_generation_pert_sim += [torch.cosine_similarity(torch.mean(hidden_sates_perturbed[-1].detach(), axis=1), latent_generation_unpert) ]

    if classifier is not None:
      discrim_losses.append(discrim_loss.data.cpu().numpy())
    losses_in_time.append(loss_in_time)

  if device == 'cuda':
    torch.cuda.empty_cache()
  pert_gen_tok_texts = [texts for _,texts in sorted(zip(latent_generation_pert_sim,pert_gen_tok_texts),reverse=True)]
  return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time

def run_pplm(
        pretrained_model="gpt2-medium",
        model_mode="online",
        cond_text="",
        uncond=False,
        num_samples=1,
        discrim=None,
        class_label=-1,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=0,
        no_cuda=False,
        verbosity='regular',
        semantic_weight = 0.2
):

    """

        Generator of Guided Language Modeling

        pretrained_model (str) : Pretrained unconditional language Modeling
        cond_text (str) : Preconditional text for generating text
        uncond (bool) : Set generator to generate from end-of-text as prefix
        num_samples (int): Number of samples to generate from the modified latents
        discrim (str): Discriminator model for conditioning langage modeling
        class_label (str): Class label used for the discriminator
        length (int): Length of generated text,
        stepsize (float): Step size for updating latent representation with gradient (the learning rate from always)
        temperature (float): for predicted logit values  
        top_k (int): Top k for beam searching
        sample (bool): if sample==1: Sample from logits distribution on generator else Sample top probable token,
        num_iterations (int): Number of iterations for lantent updating
        grad_length (int)
        horizon_length (int): Length of future to optimize over
        window_length (int): Window length to update latent representation
        decay (bool): decay for window update of latent
        gamma (float) : Scaling coefficient for the normalization term
        gm_scale (float): [0, 1] for scaling logits from perturbed and unperturbed model combination
        kl_scale (float): Kullback–Leibler scaling coeficient
        seed (int): Seed for random
        no_cuda (bool): if no_cuda == True: not to use cuda aceleration,
        verbosity(str)

    """

    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    if discrim is not None:
        discriminator_pretrained_model = paramspplm.DISCRIMINATOR_MODELS_PARAMS[discrim][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model
            if verbosity_level >= REGULAR:
                print("discrim = {}, pretrained_model set "
                "to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model

    prefix = 'data' if model_mode == 'offline' else ''

    model = GPT2LMHeadModel.from_pretrained(
        os.path.join(prefix , pretrained_model),
        output_hidden_states=True,
        use_cache=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(prefix , pretrained_model))

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    if uncond:
        tokenized_cond_text = tokenizer.encode(
            [tokenizer.bos_token],
            add_special_tokens=False
        )
    else:
        raw_text = cond_text
        if not raw_text:
              print(f"{bcolors.FAIL}{bcolors.BOLD}Insert conditional text{bcolors.ENDC}")
              exit(1)
        tokenized_cond_text = tokenizer.encode(
            tokenizer.bos_token + raw_text,
            add_special_tokens=False
        )

    print("= Prefix of sentence =")
    print(tokenizer.decode(tokenized_cond_text))
    print()

    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
        num_samples=num_samples,
        discrim=discrim,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity_level=verbosity_level,
        semantic_weight=semantic_weight
    )

    # untokenize unperturbed text
    unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])
    eot = unpert_gen_text.rfind('<|endoftext|>')
    unpert_gen_text = unpert_gen_text[: len(unpert_gen_text) if not eot else eot]
    eot = unpert_gen_text.rfind('.')
    unpert_gen_text = unpert_gen_text[: len(unpert_gen_text) if eot == -1 else eot+1]


    if verbosity_level >= REGULAR:
        print(f"{bcolors.OKCYAN}{bcolors.BOLD}{'=' * 80}{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}= Unperturbed generated text ={bcolors.ENDC}")
    print(unpert_gen_text.replace('<|endoftext|>', ''))
    print()

    generated_texts = []

    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize unperturbed text
            pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])
            
            eot = pert_gen_text.rfind('<|endoftext|>')
            pert_gen_text = pert_gen_text[: len(pert_gen_text) if not eot else eot]
            eot = pert_gen_text.rfind('.')
            pert_gen_text = pert_gen_text[: len(pert_gen_text) if eot == -1 else eot+1]
            pert_gen_text = pert_gen_text.replace('<|endoftext|>', '')
            

            resonance = getResonanceInfo(pert_gen_text)
            resonance = ' '.join([f'{f}: {i}' for f, i in zip('OCEAN', resonance)])

            print(f"{bcolors.OKCYAN}{bcolors.BOLD}= Perturbed generated text{i+1}  {resonance}={bcolors.ENDC}")
            print(pert_gen_text, end='\n\n')
        except:
            pass

        # keep the prefix, perturbed seq, original seq for each index
        generated_texts.append(
            (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
        )
    return


if __name__ == '__main__':
    pass