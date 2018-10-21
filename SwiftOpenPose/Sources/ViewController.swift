import UIKit
import CoreML
import Vision
import Upsurge
//import CoreMedia
import AVFoundation
import MobileCoreServices

class ViewController: UIViewController,AVCaptureVideoDataOutputSampleBufferDelegate,AVCaptureAudioDataOutputSampleBufferDelegate {
    
    //    let model = coco_pose_368()
    let model = MobileOpenPose()
    let ImageWidth = 368
    let ImageHeight = 368
    
    @IBOutlet weak var imageView: UIImageView!
    
    func onCompletion(b : Bool){
        
    }
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        
        let captureSession = AVCaptureSession()
        captureSession.beginConfiguration()
        
        let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                  for: .video, position: .unspecified)
        AVCaptureDevice.requestAccess(for: AVMediaType.video, completionHandler: onCompletion)
        guard
            let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice!),
            captureSession.canAddInput(videoDeviceInput)
            else { return }
        captureSession.addInput(videoDeviceInput)

        //let photoOutput = AVCapturePhotoOutput()
        let videoFrame = AVCaptureVideoDataOutput()
        let queue = DispatchQueue.global() //DispatchQueue(label: "com.ncy.videoQueue")
        videoFrame.setSampleBufferDelegate(self, queue: queue)
        guard captureSession.canAddOutput(videoFrame) else {return}
        captureSession.addOutput(videoFrame);
        captureSession.sessionPreset = .photo
//        captureSession.addOutput(photoOutput)
        captureSession.commitConfiguration()
        
        
        captureSession.startRunning()
        
        NSLog("$$ view did loaded ...")
    }
 
    
    
     public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection){
        NSLog("$$ captureOutput")
        let imageData = AVCapturePhotoOutput.jpegPhotoDataRepresentation(forJPEGSampleBuffer: sampleBuffer, previewPhotoSampleBuffer: nil)
        //let imageData = AVCaptureStillImageOutput.jpegStillImageNSDataRepresentation(sampleBuffer)
        let dataProvider = CGDataProvider(data: imageData as! CFData)
        let cgImageRef = CGImage(jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true, intent: CGColorRenderingIntent.defaultIntent)
        //var image = UIImage(CGImage: cgImageRef, scale: 1.0, orientation: UIImageOrientation.Right)
        
        //let fname = "hadou.jpg"
        //        let fname = "person1.jpg"
        
        let image = UIImage(cgImage: cgImageRef!, scale: 1.0, orientation: UIImageOrientation.right)
        
        print(measure(runCoreML(image)).duration)
        
    }
 
    
    func measure <T> (_ f: @autoclosure () -> T) -> (result: T, duration: String) {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = f()
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        return (result, "Elapsed time is \(timeElapsed) seconds.")
    }
    
    func runJsonFile(_ image: UIImage) {
        //        imageView.image = image
        
        let url = Bundle.main.url(forResource: "hadou", withExtension: "bin")!
        let text2 = try? String(contentsOf: url, encoding: .utf8)
        let personalData: Data = text2!.data(using: String.Encoding.utf8)!
        let json = try? JSONSerialization.jsonObject(with: personalData, options: [])
        
        if let array = json as? [Double] {
            
            var m: Array<Double> = Array()
            for i in 0..<array.count {
                m.append(Double(array[i]))
            }
            
            drawLine(m)
        }
    }
    
    lazy var classificationRequest: [VNRequest] = {
        do {
            let model = try VNCoreMLModel(for: self.model.model)
            let classificationRequest = VNCoreMLRequest(model: model, completionHandler: self.handleClassification)
            return [ classificationRequest ]
        } catch {
            fatalError("Can't load Vision ML model: \(error)")
        }
    }()
    
    func handleClassification(request: VNRequest, error: Error?) {
        
        guard let observations = request.results as? [VNCoreMLFeatureValueObservation] else { fatalError() }
        let mlarray = observations[0].featureValue.multiArrayValue!
        let length = mlarray.count
        let doublePtr =  mlarray.dataPointer.bindMemory(to: Double.self, capacity: length)
        let doubleBuffer = UnsafeBufferPointer(start: doublePtr, count: length)
        let mm = Array(doubleBuffer)
        
        drawLine(mm)
    }
    
    func runCoreML(_ image: UIImage) {
        imageView.image = image
        
        let img = image.resize(to: CGSize(width: ImageWidth,height: ImageHeight)).cgImage!
        let classifierRequestHandler = VNImageRequestHandler(cgImage: img, options: [:])
        do {
            try classifierRequestHandler.perform(self.classificationRequest)
        } catch {
            print(error)
        }
    }
    
    func drawLine(_ mm: Array<Double>){
        
        let com = PoseEstimator(ImageWidth,ImageHeight)
        
        let res = measure(com.estimate(mm))
        let humans = res.result;
        print("estimate \(res.duration)")
        
        var keypoint = [Int32]()
        var pos = [CGPoint]()
        var rhY = CGFloat(-1);
        var rsY = CGFloat(-1);
        for human in humans {
            var centers = [Int: CGPoint]()
            for i in 0...CocoPart.Background.rawValue {
                if human.bodyParts.keys.index(of: i) == nil {
                    continue
                }
                let bodyPart = human.bodyParts[i]!
                centers[i] = CGPoint(x: bodyPart.x, y: bodyPart.y)
                //                centers[i] = CGPoint(x: Int(bodyPart.x * CGFloat(imageW) + 0.5), y: Int(bodyPart.y * CGFloat(imageH) + 0.5))
                //TODO Assign y
                if(i==2){
                    rsY = bodyPart.y
                }
                if(i==4){
                    rhY = bodyPart.y
                }
            }
            
            //TODO alert
            if(rhY > CGFloat(0) && rsY > CGFloat(0) && rhY < rsY){
                //alert
            }
            
            for (pairOrder, (pair1,pair2)) in CocoPairsRender.enumerated() {
                
                if human.bodyParts.keys.index(of: pair1) == nil || human.bodyParts.keys.index(of: pair2) == nil {
                    continue
                }
                if centers.index(forKey: pair1) != nil && centers.index(forKey: pair2) != nil{
                    keypoint.append(Int32(pairOrder))
                    pos.append(centers[pair1]!)
                    pos.append(centers[pair2]!)
                    //                    addLine(fromPoint: centers[pair1]!, toPoint: centers[pair2]!, color: CocoColors[pairOrder])
                }
            }
        }
        let opencv = OpenCVWrapper()
        let layer = CALayer()
        let uiImage = opencv.renderKeyPoint(imageView.bounds, keypoint: &keypoint, keypoint_size: Int32(keypoint.count), pos: &pos)
        
        layer.frame = imageView.bounds
        layer.contents = uiImage?.cgImage
        layer.opacity = 0.6
        layer.masksToBounds = true
        self.view.layer.addSublayer(layer)
        
    }
    
    func addLine(fromPoint start: CGPoint, toPoint end:CGPoint, color: UIColor) {
        let line = CAShapeLayer()
        let linePath = UIBezierPath()
        linePath.move(to: start)
        linePath.addLine(to: end)
        line.path = linePath.cgPath
        line.strokeColor = color.cgColor
        line.lineWidth = 3
        line.lineJoin = kCALineJoinRound
        self.view.layer.addSublayer(line)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
}

extension UIImage {
    func resize(to newSize: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: newSize.width, height: newSize.height), true, 1.0)
        self.draw(in: CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        return resizedImage
    }
}

