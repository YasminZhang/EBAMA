import abc

import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List

from diffusers.models.cross_attention import CrossAttention
from torch import nn, einsum
from einops import rearrange, repeat

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True,
                save_path = None) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if save_path is not None:
        pil_img.save(save_path)
    if display_image:
        display(pil_img)
    return pil_img


class AttendExciteCrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class AttendExciteCrossAttnProcessor___:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
    
    def multi_qkv(self, q, uc_context, context_k, context_v, mask, attn):
        h = attn.heads
        
   
        true_bs = context_k[0].size(0) * h


        k_c = [attn.to_k(c_k) for c_k in context_k]
        v_c = [attn.to_v(c_v) for c_v in context_v]
        
        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)

        

        k_c  = [rearrange(k, 'b n (h d) -> (b h) n d', h=h) for k in k_c] # NOTE: modification point
        v_c  = [rearrange(v, 'b n (h d) -> (b h) n d', h=h) for v in v_c]

        sim_c  = [einsum('b i d, b j d -> b i j', q, k)  for k in k_c]


        attn_c  = [sim.softmax(dim=-1) for sim in sim_c]

        self.attnstore(attn_c[0], True, self.place_in_unet)


        
        # get c output        
        n_keys, n_values = len(k_c), len(v_c)
        if n_keys == n_values:
            out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn, v in zip(attn_c, v_c)]) / len(v_c)
        else:
            assert n_keys == 1 or n_values == 1
            out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn in attn_c for v in v_c]) / (n_keys * n_values)

        out = out_c
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)  

        return out

        


    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)

        res2 = hidden_states.shape[1] 
        # res2 = 256

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states


        if not is_cross:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)

            self.attnstore(attention_probs, is_cross, self.place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)

            hidden_states = attn.batch_to_head_dim(hidden_states)

            
        else:
            assert isinstance(encoder_hidden_states, dict), f"{type(encoder_hidden_states)}:, {encoder_hidden_states}"
            context = encoder_hidden_states
            context_k, context_v = context['k'], context['v']
            
            if res2 == 256:
                hidden_states = self.multi_qkv(query, None, context_k, context_v, attention_mask, attn)
            else:
                key = attn.to_k(context_k[0]) # TODO
                value = attn.to_v(context_v[0])

                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                attention_probs = attn.get_attention_scores(query, key, attention_mask)

                self.attnstore(attention_probs, is_cross, self.place_in_unet)

                hidden_states = torch.bmm(attention_probs, value)

                hidden_states = attn.batch_to_head_dim(hidden_states)


            

        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

               

        return hidden_states
    
class AttendExciteCrossAttnProcessor__:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
    
    def multi_qkv(self, q, uc_context, context_k, context_v, mask, attn):
        h = attn.heads
        
   
        true_bs = context_k[0].size(0) * h


        k_c = [attn.to_k(c_k) for c_k in context_k]
        v_c = [attn.to_v(c_v) for c_v in context_v]
        
        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)

        

        k_c  = [rearrange(k, 'b n (h d) -> (b h) n d', h=h) for k in k_c] # NOTE: modification point
        v_c  = [rearrange(v, 'b n (h d) -> (b h) n d', h=h) for v in v_c]

        sim_c  = [einsum('b i d, b j d -> b i j', q, k)  for k in k_c]


        attn_c  = [sim.softmax(dim=-1) for sim in sim_c]

        self.attnstore(attn_c[0], True, self.place_in_unet)


        
        # get c output        
        n_keys, n_values = len(k_c), len(v_c)
        if n_keys == n_values:
            out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn, v in zip(attn_c, v_c)]) / len(v_c)
        else:
            assert n_keys == 1 or n_values == 1
            out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn in attn_c for v in v_c]) / (n_keys * n_values)

        out = out_c
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)  

        return out

        


    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)

        res2 = hidden_states.shape[1] 
        # res2 = 256

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states


        if not is_cross:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)

            self.attnstore(attention_probs, is_cross, self.place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)

            hidden_states = attn.batch_to_head_dim(hidden_states)

            
        else:
            assert isinstance(encoder_hidden_states, dict), f"{type(encoder_hidden_states)}:, {encoder_hidden_states}"
            context = encoder_hidden_states
            context_k, context_v = context['k'], context['v']
            
            if res2 == 256:
                hidden_states = self.multi_qkv(query, None, context_k, context_v, attention_mask, attn)
            else:
                key = attn.to_k(context_k[0]) # TODO
                value = attn.to_v(context_v[0])

                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                attention_probs = attn.get_attention_scores(query, key, attention_mask)

                self.attnstore(attention_probs, is_cross, self.place_in_unet)

                hidden_states = torch.bmm(attention_probs, value)

                hidden_states = attn.batch_to_head_dim(hidden_states)


            

        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

               

        return hidden_states
     
    
class AttendExciteCrossAttnProcessor_:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
    
    def multi_qkv(self, q, uc_context, context_k, context_v, mask, attn):
        h = attn.heads

        assert uc_context.size(0) == context_k[0].size(0) == context_v[0].size(0)
        true_bs = uc_context.size(0) * h

        k_uc, v_uc = attn.to_k(uc_context), attn.to_v(uc_context)
        k_c = [attn.to_k(c_k) for c_k in context_k]
        v_c = [attn.to_v(c_v) for c_v in context_v]
        
        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)

        k_uc = rearrange(k_uc, 'b n (h d) -> (b h) n d', h=h)            
        v_uc = rearrange(v_uc, 'b n (h d) -> (b h) n d', h=h)

        k_c  = [rearrange(k, 'b n (h d) -> (b h) n d', h=h) for k in k_c] # NOTE: modification point
        v_c  = [rearrange(v, 'b n (h d) -> (b h) n d', h=h) for v in v_c]

        # get composition
        sim_uc = einsum('b i d, b j d -> b i j', q[:true_bs], k_uc)  
        sim_c  = [einsum('b i d, b j d -> b i j', q[true_bs:], k)  for k in k_c]

        attn_uc = sim_uc.softmax(dim=-1)
        attn_c  = [sim.softmax(dim=-1) for sim in sim_c]

        # self.attnstore(attention_probs, is_cross, self.place_in_unet) # TODO: store attn maps


        # if self.save_map and sim_uc.size(1) != sim_uc.size(2):
        #     self.save_attn_maps(attn_c)

        # get uc output
        out_uc = einsum('b i j, b j d -> b i d', attn_uc, v_uc)
        
        # get c output        
        n_keys, n_values = len(k_c), len(v_c)
        if n_keys == n_values:
            out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn, v in zip(attn_c, v_c)]) / len(v_c)
        else:
            assert n_keys == 1 or n_values == 1
            out_c = sum([einsum('b i j, b j d -> b i d', attn, v) for attn in attn_c for v in v_c]) / (n_keys * n_values)

        out = torch.cat([out_uc, out_c], dim=0)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)  

        return out

        


    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)

        res2 = hidden_states.shape[1] 
        # res2 = 256

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states


        if not is_cross or not isinstance(encoder_hidden_states, list): 
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)

            self.attnstore(attention_probs, is_cross, self.place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)

            hidden_states = attn.batch_to_head_dim(hidden_states)

            
        else:
            context = encoder_hidden_states
            assert isinstance(context, list), f"{type(context)}:, {context}"
            uc_context = context[0]
            context_k, context_v = context[1]['k'], context[1]['v']
            if res2 == 256:
                hidden_states = self.multi_qkv(query, uc_context, context_k, context_v, attention_mask, attn)
            else:
                key = attn.to_k(torch.cat((uc_context, context_k[0]), 0)) # TODO
                value = attn.to_v(torch.cat((uc_context, context_v[0]), 0))

                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                attention_probs = attn.get_attention_scores(query, key, attention_mask)

                self.attnstore(attention_probs, is_cross, self.place_in_unet)

                hidden_states = torch.bmm(attention_probs, value)

                hidden_states = attn.batch_to_head_dim(hidden_states)


            

        
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

               

        return hidden_states
    
def register_attention_control_(model, controller):

    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteCrossAttnProcessor_(
            attnstore=controller, place_in_unet=place_in_unet
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count



def register_attention_control(model, controller):

    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteCrossAttnProcessor__(
            attnstore=controller, place_in_unet=place_in_unet
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)

        if is_cross:
            a = 1
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super().reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super().__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0


class AttentionRetain(AttentionStore):
    """
    Replace the attn maps for the attribute tokens with the attn maps for the object tokens 
    """
    def __init__(self, num_steps=50,
                 cross_replace_steps=0.1,
                 association: List[Tuple[str, str]]=None):
        # execute parent class's __init__ function
        super().__init__()
        if type(cross_replace_steps) is float:
            cross_replace_steps = 0, cross_replace_steps
        self.num_cross_replace = int(num_steps * cross_replace_steps[0]), int(num_steps * cross_replace_steps[1])
        self.association = association
        self.batch_size = 1


    def forward(self, attn, is_cross: bool, place_in_unet: str):

        super().forward(attn, is_cross, place_in_unet) # store the attention maps
        if (is_cross) and (self.num_cross_replace[0] <= self.cur_step < self.num_cross_replace[1]): 
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])

            if attn.shape[2] == 256:
                temp_attn = None
                masks_object = []
                for ids in self.association:
                    
                    adj_id, object_id = ids
                    # attn[0, :, :, adj_id] = attn[0, :, :, object_id]
                    #print('attn shape:', attn.shape) # torch.Size([1, 16, res**2, 77])

                    th = attn[0, :, :, object_id].mean() + attn[0, :, :, object_id].std()

                    print(f'max value: {attn.max(1)[0].max(1)[0][0][:10]}')
    
                    masks_object.append(attn[0, :, :, object_id] > th)
                    temp_attn = attn[0, :, :, adj_id] if temp_attn is None else temp_attn + attn[0, :, :, adj_id]

                for k, ids in enumerate(self.association):
                    adj_id, object_id = ids
                    #print('mean:', masks_object[k].sum()/masks_object[k].numel())
                    print(masks_object[k].shape)
                    attn[0, :, :, adj_id] = temp_attn * masks_object[k]

            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
class AttentionReweight(AttentionStore):
    """
    Replace the attn maps for the attribute tokens with the attn maps for the object tokens 
    """
    def __init__(self, num_steps=50,
                 cross_replace_steps=0.1,
                 tokens: List[Tuple[str, str]]=None, c=2):
        # execute parent class's __init__ function
        super().__init__()
        if type(cross_replace_steps) is float:
            cross_replace_steps = 0, cross_replace_steps
        self.num_cross_replace = int(num_steps * cross_replace_steps[0]), int(num_steps * cross_replace_steps[1])
        self.tokens = tokens
        self.batch_size = 1
        self.c = c


    def forward(self, attn, is_cross: bool, place_in_unet: str):

        super().forward(attn, is_cross, place_in_unet) # store the attention maps
        if (is_cross) and (self.num_cross_replace[0] <= self.cur_step < self.num_cross_replace[1]): 
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
        
            for id in self.tokens:
                attn[0, :, :, id] *= self.c
            
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
     
class AttentionStructured(AttentionStore):
    def __init__(self, save_global_store=False):
        super().__init__(save_global_store)


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out
