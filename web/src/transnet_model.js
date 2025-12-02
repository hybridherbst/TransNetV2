
import { PreTrainedModel, Tensor } from '@huggingface/transformers';

export class TransNetV2Model extends PreTrainedModel {
    async _call(inputs) {
        // TransNetV2 expects an input named 'input'
        // inputs can be { input: tensor } or just tensor if we handle it
        if (this.sessions && this.sessions.model) {
            return await this.sessions.model.run(inputs);
        } else {
            console.log('Available properties:', Object.keys(this));
            throw new Error("No session found in TransNetV2Model (checked this.sessions.model)");
        }
    }
}
