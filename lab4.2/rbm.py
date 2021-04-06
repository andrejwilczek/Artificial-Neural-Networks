from util import *


class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''

    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28, 28], is_top=False, n_labels=10, batch_size=10):
        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """

        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom:
            self.image_size = image_size

        self.is_top = is_top

        if is_top:
            self.n_labels = 10

        self.batch_size = batch_size

        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(
            loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(
            loc=0.0, scale=0.01, size=(self.ndim_visible, self.ndim_hidden))

        self.bias_h = np.random.normal(
            loc=0.0, scale=0.01, size=(self.ndim_hidden))

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0

        self.weight_v_to_h = None

        self.weight_h_to_v = None

        self.learning_rate = 1/batch_size

        self.momentum = 0.7

        self.print_period = 1

        self.rf = {  # receptive-fields. Only applicable when visible layer is input data
            "period": 5000,  # iteration period to visualize
            "grid": [5, 5],  # size of the grid
            # pick some random hidden units
            "ids": np.random.randint(0, self.ndim_hidden, 25)
        }

        return

    def cd1(self, visible_trainset, n_iterations=10000):
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("learning CD1")

        n_samples = visible_trainset.shape[0]
        n_batch = int(n_samples/self.batch_size)
        # print(n_batch)

        # [DONE TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
        # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
        # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.

        # print(self.weight_vh.shape)
        loss_vec = []
        it_vec = []
        for it in range(n_iterations):
            reconstructed_input = np.zeros(visible_trainset.shape)
            reconstructed_hidden = np.zeros(
                (visible_trainset.shape[0], self.ndim_hidden))
            for batch in range(n_batch):
                visible_minibatch = visible_trainset[batch *
                                                     self.batch_size:self.batch_size+batch*self.batch_size][:]

                # Get v0
                # temp= sigmoid(self.bias_v)
                # for b in range(self.batch_size-1):
                #     temp=np.vstack((temp,sigmoid(self.bias_v)))
                on_prob = sigmoid(self.bias_v)
                # print(on_prob.shape)

                v0 = (on_prob, visible_minibatch)
                # plt.imshow(v0[1][0][:].reshape(28, 28))
                # plt.show()

                # print(v0.shape)

                h0 = self.get_h_given_v(v0[1])
                # print(h0[1].shape)
                vk = self.get_v_given_h(h0[1])

                hk = self.get_h_given_v(vk[1])

                reconstructed_input[batch*self.batch_size:self.batch_size +
                                    batch*self.batch_size][:] = vk[1]
                reconstructed_hidden[batch*self.batch_size:self.batch_size +
                                     batch*self.batch_size][:] = hk[1]
                # print(hk_activation.shape)

            # [DONE TASK 4.1] update the parameters using function 'update_params'

                self.update_params(v0, h0, vk, hk)
                # if batch % self.figure_period == 0:

            # visualize once in a while when visible layer is input images

            if it % self.rf["period"] == 0 and self.is_bottom:

                viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape(
                    (self.image_size[0], self.image_size[1], -1)), it=it, grid=self.rf["grid"])

            # print progress
            loss = np.linalg.norm(
                visible_trainset - reconstructed_input)/visible_trainset.shape[0]
            loss_vec.append(loss)
            it_vec.append(it)
            if it % self.print_period == 0:
                print("iteration=%3d/%3d \t recon_loss=%4.4f" %
                      (it+1, n_iterations, loss))
        plt.figure('Digit Estimate')
        for i in range(9):
            plt.subplot(3, 3, i+1)
            first_im = vk[1][i][:]
            first_im = first_im.reshape(28, 28)
            plt.imshow(first_im)
        plt.show()
        for i in range(9):
            plt.subplot(3, 3, i+1)
            first_im = v0[1][i][:]
            first_im = first_im.reshape(28, 28)
            plt.imshow(first_im)
        plt.show()
        return reconstructed_hidden

    def update_params(self, v_0, h_0, v_k, h_k):
        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

       # [DONE TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters

        self.delta_bias_v = np.sum(self.learning_rate*(v_0[0]-v_k[1]), axis=0)
        self.delta_weight_vh = self.learning_rate * \
            (np.dot(np.transpose(v_0[0]), h_0[0]) -
             np.dot(np.transpose(v_k[0]), h_k[1]))
        self.delta_bias_h = np.sum(self.learning_rate*(h_0[0]-h_k[1]), axis=0)

        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h

        return

    def get_h_given_v(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"

        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None

        on_prob = sigmoid(
            self.bias_h + np.dot(visible_minibatch, self.weight_vh))
        activations = sample_binary(on_prob)

        return (on_prob, activations)

    def get_v_given_h(self, hidden_minibatch):
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            support = self.bias_v + \
                np.dot(hidden_minibatch, np.transpose(self.weight_vh))

            prob_hidden = sigmoid(support[:, :-self.n_labels])
            prob_lbl = softmax(support[:, -self.n_labels:])

            activations = np.hstack(
                (sample_binary(prob_hidden), sample_categorical(prob_lbl)))
            on_prob = np.hstack((prob_hidden, prob_lbl))

        else:

            on_prob = sigmoid(
                self.bias_v + np.dot(hidden_minibatch, np.transpose(self.weight_vh)))
            activations = sample_binary(on_prob)

        return (on_prob, activations)

    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self):

        self.weight_v_to_h = np.copy(self.weight_vh)
        self.weight_h_to_v = np.copy(np.transpose(self.weight_vh))
        self.weight_vh = None

    def get_h_given_v_dir(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"

        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        on_prob = sigmoid(
            self.bias_h + np.dot(visible_minibatch, self.weight_v_to_h))
        activations = sample_binary(on_prob)

        return (on_prob, activations)

    def get_v_given_h_dir(self, hidden_minibatch):
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"

        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)
            raise 'We should not be here! line: 316 (ish)'

        else:

            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)

            on_prob = sigmoid(
                self.bias_v + np.dot(hidden_minibatch, self.weight_h_to_v))
            activations = sample_binary(on_prob)

        return (on_prob, activations)

    def update_generate_params(self, inps, trgs, preds):
        """Update generative weight "weight_h_to_v" and bias "bias_v"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v

        return

    def update_recognize_params(self, inps, trgs, preds):
        """Update recognition weight "weight_v_to_h" and bias "bias_h"

        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h

        return
